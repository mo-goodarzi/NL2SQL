"""Reject any SQL that is not a single read-only SELECT.

A language model writing SQL against your database is a real injection
surface, and handling it explicitly is the kind of production-risk thinking
that separates a portfolio project from a demo. Say so in your README.

**This is the second layer, never the only one.** ``db.py`` opens connections
with ``mode=ro`` and sets ``PRAGMA query_only``, which is the actual
guarantee -- a destructive statement would fail there even if this module let
it through. What this module adds is a fast, legible rejection *with a reason
the agent can act on*, before a query is ever sent to the database. That
reason flows into the same correction loop as a database error, so an unsafe
query gets rewritten rather than simply refused.

The hard part is not blocking bad SQL; it is not blocking good SQL. A naive
keyword scan rejects ``SELECT REPLACE(city, 'a', 'b')`` because it contains
REPLACE, and ``WHERE comment = 'DROP TABLE'`` because a string literal happens
to contain DROP. Every false rejection burns a retry and depresses your
accuracy for a reason that has nothing to do with the model's SQL ability.
So the scan runs against a sanitised copy with comments, string literals, and
quoted identifiers removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Statement types that must never reach the database.
#
# Word-boundary matched against sanitised SQL, so a column named `created_at`
# does not trip CREATE, and `updated_by` does not trip UPDATE.
# ---------------------------------------------------------------------------

FORBIDDEN_KEYWORDS: frozenset[str] = frozenset({
    # data modification
    "INSERT", "UPDATE", "DELETE", "TRUNCATE", "UPSERT",
    # schema modification
    "CREATE", "ALTER", "DROP", "RENAME",
    # database and file access -- ATTACH is the serious one: it can read or
    # create database files outside the one we intend to expose.
    "ATTACH", "DETACH", "VACUUM", "REINDEX",
    # engine configuration
    "PRAGMA",
    # transaction control -- no legitimate use in a generated read query
    "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE",
    # trigger/permission statements (not all exist in SQLite; blocked anyway
    # so the same module survives a move to Postgres)
    "GRANT", "REVOKE", "EXECUTE", "CALL", "MERGE",
})

#: REPLACE needs special handling: `REPLACE INTO` is a write, but REPLACE() is
#: an ordinary scalar function that legitimately appears in SELECT clauses.
#: Blocking the bare keyword would reject correct queries.
_REPLACE_INTO = re.compile(r"\bREPLACE\s+INTO\b", re.IGNORECASE)

#: Statement must begin with one of these.
_ALLOWED_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)

#: SQLite function that can load arbitrary code. Not available through the
#: Python driver by default, blocked regardless.
_DANGEROUS_FUNCTIONS = re.compile(
    r"\b(load_extension|readfile|writefile|edit|fts3_tokenizer)\s*\(",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Validation:
    """Outcome of validating one statement."""

    ok: bool
    #: Written for the model, not just the log: it is fed back into the
    #: correction prompt, so it says what to do rather than only what is wrong.
    reason: str | None = None
    #: The comment- and literal-stripped text the checks ran against. Useful
    #: when a rejection looks wrong and you need to see what was scanned.
    sanitized: str | None = None

    def __bool__(self) -> bool:
        return self.ok


def strip_sql(sql: str) -> str:
    """Remove comments, string literals, and quoted identifiers.

    Returns text of the same broad structure with those regions blanked, so
    keyword and semicolon scanning cannot be fooled by their contents. The
    result is not valid SQL and is never executed -- it exists only to be
    searched.

    Handles: ``--`` line comments, ``/* */`` block comments, single-quoted
    strings with ``''`` escapes, double-quoted and backtick-quoted
    identifiers, and ``[bracketed]`` identifiers.
    """
    out: list[str] = []
    i, n = 0, len(sql)

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        # -- line comment
        if ch == "-" and nxt == "-":
            j = sql.find("\n", i)
            i = n if j == -1 else j
            out.append(" ")
            continue

        # /* block comment */  (unterminated: treat rest as comment)
        if ch == "/" and nxt == "*":
            j = sql.find("*/", i + 2)
            i = n if j == -1 else j + 2
            out.append(" ")
            continue

        # 'string literal' with '' escape
        if ch == "'":
            i += 1
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append("''")
            continue

        # "identifier" / `identifier` / [identifier]
        if ch in ('"', "`", "["):
            closing = "]" if ch == "[" else ch
            j = sql.find(closing, i + 1)
            i = n if j == -1 else j + 1
            out.append(" id ")
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _split_statements(sanitized: str) -> list[str]:
    """Split on semicolons that are real statement separators.

    Literals and comments are already blanked, so any remaining semicolon is
    structural. A single trailing semicolon is not a second statement.
    """
    return [s for s in (part.strip() for part in sanitized.split(";")) if s]


def validate_sql(sql: str) -> Validation:
    """Check that ``sql`` is a single read-only SELECT. Never raises.

    Returns a ``Validation``; on failure ``reason`` is phrased as an
    instruction so it can be fed directly into the correction prompt and
    produce a usable retry rather than a repeat of the same mistake.
    """
    if sql is None or not sql.strip():
        return Validation(False, "No SQL was provided. Return a SELECT query.")

    sanitized = strip_sql(sql)
    statements = _split_statements(sanitized)

    if not statements:
        return Validation(
            False,
            "The statement contains no SQL, only comments. Return a SELECT query.",
            sanitized,
        )

    # --- one statement only ------------------------------------------------
    # Checked before the keyword scan so a stacked query gets the specific,
    # more actionable message.
    if len(statements) > 1:
        return Validation(
            False,
            f"Only one statement is allowed, but {len(statements)} were found. "
            "Combine the logic into a single SELECT query.",
            sanitized,
        )

    statement = statements[0]

    # --- must start with SELECT or WITH ------------------------------------
    if not _ALLOWED_START.match(statement):
        first = (re.match(r"\s*(\w+)", statement) or [None, "?"])[1].upper()
        return Validation(
            False,
            f"Only read-only queries are allowed, but the statement begins with "
            f"{first}. Rewrite it as a SELECT (a WITH clause is also allowed).",
            sanitized,
        )

    # --- no write or admin statements anywhere -----------------------------
    # Catches them inside subqueries and CTEs, not just at the start.
    upper = statement.upper()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return Validation(
                False,
                f"The keyword {kw} is not permitted; this database is read-only. "
                "Rewrite the query to only read data with SELECT.",
                sanitized,
            )

    if _REPLACE_INTO.search(statement):
        return Validation(
            False,
            "REPLACE INTO writes data and is not permitted; this database is "
            "read-only.",
            sanitized,
        )

    if _DANGEROUS_FUNCTIONS.search(statement):
        return Validation(
            False,
            "That function is not permitted. Use only standard SQL functions "
            "for querying data.",
            sanitized,
        )

    return Validation(True, None, sanitized)


def assert_safe(sql: str) -> str:
    """Return ``sql`` if valid, else raise ``ValueError``.

    For trusted call sites that genuinely want an exception. The agent should
    use ``validate_sql`` instead: it needs the rejection as a value it can put
    into a prompt, not as an exception that unwinds the graph.
    """
    result = validate_sql(sql)
    if not result.ok:
        raise ValueError(result.reason)
    return sql


if __name__ == "__main__":  # python -m nl2sql.guardrails
    samples = [
        "SELECT * FROM orders",
        "WITH x AS (SELECT 1 AS n) SELECT n FROM x",
        "SELECT REPLACE(customer_city, 'sao', 'SAO') FROM customers",
        "SELECT * FROM orders WHERE order_status = 'canceled'; DROP TABLE orders",
        "DELETE FROM orders",
        "SELECT * FROM orders -- ; DROP TABLE orders",
        "PRAGMA table_info(orders)",
    ]
    for s in samples:
        v = validate_sql(s)
        print(f"{'PASS' if v.ok else 'FAIL'}  {s}")
        if not v.ok:
            print(f"      {v.reason}")