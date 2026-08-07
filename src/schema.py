"""Introspect the database and decide how much schema the model sees.

How schema is presented is the highest-leverage variable in text-to-SQL
accuracy, which is why this gets its own module rather than living inside a
prompt template. Two modes:

* **Full** (baseline behaviour) -- every table, every column.
* **Retrieval** (Step 4 improvement) -- only the tables relevant to the
  question, which shortens the prompt, removes distractors, and reuses the
  retrieval skills from Project 1.

**The trap in retrieval mode**, and the reason this module is longer than it
first appears: selecting tables by relevance alone produces schemas that
cannot be joined. "Revenue by product category" scores highly on ``products``
and ``order_items``, and "average review score per category" scores highly on
``order_reviews`` and ``products`` -- but the second pair cannot be joined
without ``orders`` and ``order_items`` in between. Hand the model a schema with
a missing bridge table and it will either invent a join or give up, and your
accuracy lands *below* the full-schema baseline. So retrieval always runs a
foreign-key expansion pass that adds whatever tables are needed to connect
the selected ones.

Ranking is lexical by default: no extra dependency, no API cost, and fully
deterministic, which matters because a non-deterministic retriever makes
evaluation runs incomparable. An embedding backend is available for when you
want to say "semantic retrieval" and mean it.
"""

from __future__ import annotations

import difflib
import math
import re
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Sequence

from config import settings
from db import execute_query, fetch_all

# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Column:
    name: str
    type: str
    not_null: bool
    primary_key: bool


@dataclass(frozen=True)
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


@lru_cache(maxsize=1)
def get_tables() -> dict[str, Table]:
    """Introspect every user table. Cached -- the schema does not change."""
    names = [
        r[0] for r in fetch_all(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    tables: dict[str, Table] = {}
    for name in names:
        cols = tuple(
            Column(name=r[1], type=(r[2] or "").upper(), not_null=bool(r[3]),
                   primary_key=bool(r[5]))
            for r in fetch_all(f"PRAGMA table_info({name})")
        )
        fks = tuple(
            ForeignKey(column=r[3], ref_table=r[2], ref_column=r[4])
            for r in fetch_all(f"PRAGMA foreign_key_list({name})")
        )
        tables[name] = Table(name=name, columns=cols, foreign_keys=fks)
    return tables


def get_table_names() -> list[str]:
    return list(get_tables())


def describe_table(name: str) -> str:
    """Human-readable description of one table, for the agent's schema tool."""
    tables = get_tables()
    if name not in tables:
        close = difflib.get_close_matches(name, tables, n=3, cutoff=0.5)
        hint = f" Did you mean: {', '.join(close)}?" if close else \
               f" Available tables: {', '.join(tables)}."
        return f"No table named '{name}'.{hint}"
    return render_table(tables[name], include_samples=settings.include_sample_values)


# ---------------------------------------------------------------------------
# Sample values
# ---------------------------------------------------------------------------

#: Columns whose sample values are useless and expensive: opaque identifiers.
_ID_LIKE = re.compile(r"(^|_)(id|uuid|hash)$", re.IGNORECASE)


def _is_worth_sampling(col: Column) -> bool:
    """Sample only low-cardinality text columns.

    Sampling ``order_id`` shows the model 32-character hashes it can never use.
    Sampling ``order_status`` shows it that values are lower-case, which is
    worth real accuracy on filter questions.
    """
    if _ID_LIKE.search(col.name) or col.primary_key:
        return False
    return col.type in {"TEXT", "VARCHAR", "CHAR", ""}


@lru_cache(maxsize=256)
def _sample_values(table: str, column: str, n: int) -> tuple[str, ...]:
    """A few distinct values, read from a bounded scan.

    The subquery LIMIT matters: ``SELECT DISTINCT`` over a million-row table
    scans the lot. Sampling the first 500 rows is fast and, for the
    low-cardinality columns we bother with, representative.
    """
    result = execute_query(
        f"SELECT DISTINCT {column} FROM "
        f"(SELECT {column} FROM {table} LIMIT 500) "
        f"WHERE {column} IS NOT NULL LIMIT {n}",
        max_rows=n,
        timeout_s=2.0,
    )
    if not result.ok:
        return ()
    out = []
    for row in result.rows:
        v = str(row[0])
        if len(v) > 30:  # long free text adds noise, not signal
            return ()
        out.append(v)
    return tuple(out)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_table(table: Table, include_samples: bool = True) -> str:
    """Render one table as compact CREATE TABLE-style text.

    Chosen over raw ``sqlite_master`` SQL because it is denser (models pay per
    token), and over a bare column list because the foreign keys are the part
    that tells the model how to join.
    """
    lines = [f"CREATE TABLE {table.name} ("]
    body: list[str] = []
    for col in table.columns:
        parts = [f"  {col.name} {col.type or 'TEXT'}"]
        if col.primary_key:
            parts.append("PRIMARY KEY")
        elif col.not_null:
            parts.append("NOT NULL")
        line = " ".join(parts)
        if include_samples and _is_worth_sampling(col):
            vals = _sample_values(table.name, col.name,
                                  settings.sample_values_per_column)
            if vals:
                line += "  -- e.g. " + ", ".join(repr(v) for v in vals)
        body.append(line)

    for fk in table.foreign_keys:
        body.append(f"  FOREIGN KEY ({fk.column}) "
                    f"REFERENCES {fk.ref_table}({fk.ref_column})")

    lines.append(",\n".join(body))
    lines.append(")")
    return "\n".join(lines)


def render_schema(tables: Iterable[Table], include_samples: bool = True) -> str:
    return "\n\n".join(render_table(t, include_samples) for t in tables)


def get_full_schema_text(include_samples: bool | None = None) -> str:
    """Every table. The baseline's view of the world."""
    samples = settings.include_sample_values if include_samples is None else include_samples
    return render_schema(get_tables().values(), include_samples=samples)


# ---------------------------------------------------------------------------
# Relevance ranking
# ---------------------------------------------------------------------------

#: Domain vocabulary the schema does not contain. A user says "revenue"; no
#: column is called that. Curated synonyms are cheap and move retrieval
#: accuracy more than a better ranking function would -- but keep the list
#: short and grounded in questions your eval actually contains.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "revenue": ("price", "order", "item"),
    "sales": ("price", "order", "item"),
    "spend": ("payment", "price"),
    "income": ("price", "payment"),
    "buyer": ("customer",),
    "shopper": ("customer",),
    "client": ("customer",),
    "vendor": ("seller",),
    "merchant": ("seller",),
    "rating": ("review", "score"),
    "feedback": ("review",),
    "score": ("review",),
    "category": ("product", "category", "translation"),
    "shipping": ("freight", "item"),
    "delivery": ("order", "delivered"),
    "late": ("order", "delivered", "estimated"),
    "installment": ("payment",),
    "card": ("payment",),
    "state": ("customer", "seller", "geolocation"),
    "city": ("customer", "seller", "geolocation"),
    "location": ("geolocation",),
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, crudely singularised.

    Naive plural stripping ('customers' -> 'customer') matters because table
    names are plural and questions are usually not.
    """
    toks = _TOKEN_RE.findall(text.lower())
    out = []
    for t in toks:
        out.append(t)
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            out.append(t[:-1])
    return out


def _table_tokens(table: Table) -> set[str]:
    toks = set(_tokenize(table.name))
    for col in table.columns:
        toks.update(_tokenize(col.name))
    return toks


@lru_cache(maxsize=1)
def _idf() -> dict[str, float]:
    """Inverse document frequency across tables.

    Without it, ``order_id`` -- which appears in five tables -- would drag
    every table into every result. IDF makes distinctive tokens like
    ``review_score`` or ``freight_value`` do the work.
    """
    tables = get_tables()
    n = len(tables)
    df: dict[str, int] = {}
    for t in tables.values():
        for tok in _table_tokens(t):
            df[tok] = df.get(tok, 0) + 1
    return {tok: math.log((n + 1) / (c + 0.5)) for tok, c in df.items()}


def score_tables_lexical(question: str) -> dict[str, float]:
    """IDF-weighted token overlap between question and each table."""
    idf = _idf()
    q_tokens = list(_tokenize(question))
    for tok in list(q_tokens):
        q_tokens.extend(SYNONYMS.get(tok, ()))

    scores: dict[str, float] = {}
    for name, table in get_tables().items():
        toks = _table_tokens(table)
        score = sum(idf.get(t, 0.5) for t in set(q_tokens) if t in toks)
        # An explicit table-name mention should always dominate.
        if name.lower() in question.lower():
            score += 10.0
        scores[name] = score
    return scores


def score_tables_embedding(question: str) -> dict[str, float]:
    """Cosine similarity between the question and a rendered table summary.

    Optional. Requires an embedding provider and adds a per-question API call,
    so it must earn its cost against the lexical scorer in your evaluation
    rather than being assumed better.
    """
    from llm import _get_client  # local import: optional path

    client = _get_client()
    tables = get_tables()
    docs = {
        name: f"{name}: " + ", ".join(t.column_names)
        for name, t in tables.items()
    }
    texts = [question, *docs.values()]
    resp = client.embeddings.create(model=settings.embedding_model, input=texts)
    vecs = [d.embedding for d in resp.data]
    q_vec, table_vecs = vecs[0], vecs[1:]

    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    return {name: cosine(q_vec, v) for name, v in zip(docs, table_vecs)}


# ---------------------------------------------------------------------------
# Foreign-key expansion -- the part that makes retrieval work
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _fk_graph() -> dict[str, set[str]]:
    """Undirected adjacency of tables joined by a declared foreign key.

    Undirected because a join works in either direction; the declared FK
    direction is irrelevant for reachability.
    """
    graph: dict[str, set[str]] = {name: set() for name in get_tables()}
    for name, table in get_tables().items():
        for fk in table.foreign_keys:
            if fk.ref_table in graph:
                graph[name].add(fk.ref_table)
                graph[fk.ref_table].add(name)
    return graph


def _shortest_path(a: str, b: str) -> list[str] | None:
    """BFS path between two tables through foreign keys."""
    if a == b:
        return [a]
    graph = _fk_graph()
    queue = deque([[a]])
    seen = {a}
    while queue:
        path = queue.popleft()
        for nxt in sorted(graph.get(path[-1], ())):
            if nxt in seen:
                continue
            if nxt == b:
                return path + [nxt]
            seen.add(nxt)
            queue.append(path + [nxt])
    return None


def expand_along_foreign_keys(selected: Iterable[str]) -> list[str]:
    """Add whatever tables are needed to make the selection joinable.

    Without this step retrieval actively hurts: pick ``order_reviews`` and
    ``products`` for "average review score by category" and the model gets a
    schema with no path between them. Connecting them pulls in ``orders`` and
    ``order_items``, and the query becomes writable.
    """
    chosen = list(dict.fromkeys(selected))
    if len(chosen) < 2:
        return chosen

    connected = set(chosen)
    for i, a in enumerate(chosen):
        for b in chosen[i + 1:]:
            path = _shortest_path(a, b)
            if path:
                connected.update(path)
    # Preserve original ranking order, then append the bridges.
    ordered = [t for t in chosen if t in connected]
    ordered += sorted(t for t in connected if t not in ordered)
    return ordered


# ---------------------------------------------------------------------------
# Public retrieval API
# ---------------------------------------------------------------------------

@dataclass
class Retrieval:
    """What retrieval chose, and why -- for tracing and debugging."""

    tables: list[str]
    ranked: list[tuple[str, float]] = field(default_factory=list)
    added_by_expansion: list[str] = field(default_factory=list)
    method: str = "lexical"
    fell_back: bool = False


def retrieve_tables(
    question: str,
    k: int | None = None,
    method: str | None = None,
) -> Retrieval:
    """Choose the tables relevant to a question, then make them joinable."""
    k = settings.schema_top_k if k is None else k
    method = method or getattr(settings, "schema_retrieval_method", "lexical")

    try:
        scores = (score_tables_embedding(question) if method == "embedding"
                  else score_tables_lexical(question))
    except Exception:  # noqa: BLE001 -- retrieval must never break a query
        scores = score_tables_lexical(question)
        method = "lexical (fallback)"

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top = [name for name, score in ranked[:k] if score > 0]

    # Nothing matched: fall back to the full schema rather than handing the
    # model an empty or arbitrary one. A slow correct answer beats a fast
    # wrong one.
    if not top:
        return Retrieval(
            tables=get_table_names(), ranked=ranked, method=method, fell_back=True
        )

    expanded = expand_along_foreign_keys(top)
    return Retrieval(
        tables=expanded,
        ranked=ranked,
        added_by_expansion=[t for t in expanded if t not in top],
        method=method,
    )


def get_relevant_schema_text(
    question: str,
    k: int | None = None,
    include_samples: bool | None = None,
) -> str:
    """Rendered schema for just the tables this question needs."""
    samples = settings.include_sample_values if include_samples is None else include_samples
    retrieval = retrieve_tables(question, k=k)
    tables = get_tables()
    return render_schema(
        [tables[name] for name in retrieval.tables if name in tables],
        include_samples=samples,
    )


def get_schema_text(question: str | None = None) -> str:
    """The schema the agent should use, honouring ``use_schema_retrieval``.

    The single call site for nodes.py, so flipping the Step 4 flag changes
    behaviour without touching the graph.
    """
    if settings.use_schema_retrieval and question:
        return get_relevant_schema_text(question)
    return get_full_schema_text()


def clear_caches() -> None:
    """Drop introspection caches. For tests, or after rebuilding the database."""
    get_tables.cache_clear()
    _idf.cache_clear()
    _fk_graph.cache_clear()
    _sample_values.cache_clear()