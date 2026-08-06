"""The only module allowed to touch the database.

Every query in the system -- agent, baseline, and evaluation harness -- goes
through here, so read-only enforcement and timeouts are structural rather than
something each caller has to remember.

Two entry points, and the distinction between them is the point of this module:

``execute_query``
    For **untrusted** SQL, i.e. anything a language model wrote. Never raises.
    Failures come back as a ``QueryResult`` with ``ok=False`` and the database's
    own error text in ``error``. That is not a stylistic choice: the
    self-correction loop feeds ``error`` straight back into the next prompt, so
    an exception escaping this function would break the loop entirely.

``fetch_all``
    For **trusted** SQL that this codebase wrote itself -- schema introspection,
    PRAGMA calls, the evaluation harness running its own gold queries. Raises
    normally, because a failure there is a bug in the repo, not something an
    agent should try to recover from.

Read-only is enforced three ways, deliberately overlapping: the connection URI
uses ``mode=ro``, ``PRAGMA query_only`` is set on every connection, and
``guardrails.py`` rejects non-SELECT statements before they ever arrive. The
first two are guarantees; the third is a fast, legible rejection with a reason
the agent can act on.
"""

from __future__ import annotations

import difflib
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from config import settings

# One connection per thread. SQLite connections are not safe to share across
# threads, and FastAPI serves requests on a thread pool -- a single module-level
# connection would work in development and fail intermittently in production.
_local = threading.local()


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Outcome of running one SQL statement.

    Deliberately a value object with no exceptions attached: callers branch on
    ``ok`` rather than wrapping calls in try/except.
    """

    ok: bool
    sql: str
    columns: list[str] | None = None
    rows: list[tuple] = field(default_factory=list)
    row_count: int = 0
    #: Raw database error text, enriched with a hint where one helps. This is
    #: what gets fed back to the model on a retry, so it is written to be read
    #: by a model as much as by a human.
    error: str | None = None
    #: True when ``max_rows`` cut the result short. Matters for evaluation:
    #: comparing a truncated result set against a full gold one would mark a
    #: correct query wrong, which is why the harness passes ``max_rows=None``.
    truncated: bool = False
    elapsed_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        """Succeeded but returned nothing.

        Worth distinguishing from failure. Sometimes zero rows is the right
        answer; sometimes it means the model filtered on a value that does not
        exist. The agent treats it as a soft failure worth one retry, and the
        evaluation report flags these separately.
        """
        return self.ok and self.row_count == 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for the API response and trace payloads."""
        return {
            "ok": self.ok,
            "sql": self.sql,
            "columns": self.columns,
            "rows": [list(r) for r in self.rows],
            "row_count": self.row_count,
            "error": self.error,
            "truncated": self.truncated,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }

    def preview(self, n: int = 3) -> str:
        """Short human-readable summary for logs and terminal output."""
        if not self.ok:
            return f"ERROR: {self.error}"
        if self.is_empty:
            return "0 rows"
        head = ", ".join(str(v) for v in self.rows[0][:4])
        more = f" (+{self.row_count - 1} more)" if self.row_count > 1 else ""
        return f"{self.row_count} rows | {head}{more}"


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Return this thread's read-only connection, creating it on first use."""
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is not None:
        return conn

    settings.require_database()  # actionable message if the DB isn't built yet

    conn = sqlite3.connect(
        settings.read_only_uri,
        uri=True,
        check_same_thread=False,
        # Lock-wait timeout, unrelated to our statement timeout below.
        timeout=5.0,
    )
    # Second layer of read-only enforcement, independent of the URI.
    conn.execute("PRAGMA query_only = ON")
    # Foreign keys on so introspection reports relationships consistently.
    conn.execute("PRAGMA foreign_keys = ON")
    # Third layer, and the only one that covers ATTACH: query_only and the
    # read-only URI stop writes to olist.db itself, but neither reaches a
    # *second* database a query attaches, so untrusted SQL could ATTACH an
    # arbitrary file and SELECT it straight out. Denied at the C level so no
    # SQL-text filtering in guardrails.py can be bypassed by rephrasing it.
    conn.set_authorizer(_authorize)
    _local.conn = conn
    return conn


def _authorize(
    action: int,
    arg1: str | None,
    _arg2: str | None,
    _db_name: str | None,
    _trigger_or_view: str | None,
) -> int:
    """Deny ATTACH/DETACH and PRAGMA writable_schema; allow everything else."""
    if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() == "writable_schema":
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def close_connection() -> None:
    """Close this thread's connection. Used by tests and shutdown hooks."""
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None


def _install_timeout(conn: sqlite3.Connection, seconds: float) -> None:
    """Abort long-running statements.

    SQLite has no statement timeout, so this hooks the progress handler and
    interrupts once the deadline passes. Without it, one accidental cross join
    stalls a fifty-question evaluation run indefinitely.
    """
    deadline = time.monotonic() + seconds

    def _handler() -> int:
        return 1 if time.monotonic() > deadline else 0

    # Checked every N virtual-machine instructions; 10k keeps overhead low
    # while still interrupting within a few milliseconds of the deadline.
    conn.set_progress_handler(_handler, 10_000)


def _clear_timeout(conn: sqlite3.Connection) -> None:
    conn.set_progress_handler(None, 0)


# ---------------------------------------------------------------------------
# Error enrichment
# ---------------------------------------------------------------------------

def _list_tables(conn: sqlite3.Connection) -> list[str]:
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    except sqlite3.Error:
        return []


def _list_columns(conn: sqlite3.Connection, tables: Sequence[str]) -> list[str]:
    cols: list[str] = []
    for t in tables:
        try:
            for row in conn.execute(f"PRAGMA table_info({t})").fetchall():
                cols.append(f"{t}.{row[1]}")
        except sqlite3.Error:
            continue
    return cols


def _tables_mentioned(sql: str, known: Sequence[str]) -> list[str]:
    """Which known tables appear in the SQL text.

    Crude on purpose -- a substring scan is enough to decide which columns are
    worth suggesting, and a real parser here would be cost without benefit.
    """
    low = sql.lower()
    return [t for t in known if re.search(rf"\b{re.escape(t.lower())}\b", low)]


def _enrich_error(conn: sqlite3.Connection, sql: str, message: str) -> str:
    """Add a targeted hint to a raw SQLite error.

    SQLite's messages are terse: "no such column: category_name" says what
    failed but not what would have worked. Appending the near-miss candidates
    measurably improves how often the correction loop fixes the query on its
    first retry, because the model no longer has to guess the right identifier
    from memory of the schema block.
    """
    tables = _list_tables(conn)

    m = re.search(r"no such table:\s*([\w.]+)", message)
    if m:
        bad = m.group(1).split(".")[-1]
        close = difflib.get_close_matches(bad, tables, n=3, cutoff=0.5)
        hint = f"Did you mean: {', '.join(close)}?" if close else \
               f"Available tables: {', '.join(tables)}."
        return f"{message}. {hint}"

    m = re.search(r"no such column:\s*([\w.]+)", message)
    if m:
        bad = m.group(1).split(".")[-1]
        scope = _tables_mentioned(sql, tables) or tables
        cols = _list_columns(conn, scope)
        close = difflib.get_close_matches(
            bad, [c.split(".")[1] for c in cols], n=3, cutoff=0.5
        )
        if close:
            matched = [c for c in cols if c.split(".")[1] in close]
            return f"{message}. Did you mean: {', '.join(matched)}?"
        return (
            f"{message}. Columns available in the referenced tables: "
            f"{', '.join(cols[:40])}."
        )

    if "ambiguous column name" in message:
        return f"{message}. Qualify the column with its table alias."

    return message


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_query(
    sql: str,
    max_rows: int | None = None,
    unlimited: bool = False,
    timeout_s: float | None = None,
) -> QueryResult:
    """Run untrusted (model-written) SQL. Never raises.

    Args:
        sql: The statement to run. Not validated here -- ``guardrails.py`` owns
            that, and this function stays usable for trusted introspection too.
        max_rows: Row ceiling. Defaults to ``settings.max_rows``.
        unlimited: Return every row, ignoring ``max_rows``. **The evaluation
            harness must set this**, because comparing a truncated result set
            against a full gold one would mark a correct query wrong.
        timeout_s: Statement deadline; defaults to ``settings.query_timeout_s``.

    Returns:
        A ``QueryResult``. On failure ``ok`` is False and ``error`` holds the
        database message plus any hint -- ready to be fed to the correction
        prompt verbatim.
    """
    limit = None if unlimited else (settings.max_rows if max_rows is None else max_rows)
    timeout = settings.query_timeout_s if timeout_s is None else timeout_s

    started = time.perf_counter()
    try:
        conn = get_connection()
    except (FileNotFoundError, sqlite3.Error) as exc:
        return QueryResult(
            ok=False, sql=sql, error=str(exc),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    cursor = None
    try:
        _install_timeout(conn, timeout)
        cursor = conn.execute(sql)

        # Fetch one extra row to detect truncation without a second query.
        if limit is None:
            rows = cursor.fetchall()
            truncated = False
        else:
            fetched = cursor.fetchmany(limit + 1)
            truncated = len(fetched) > limit
            rows = fetched[:limit]

        columns = [d[0] for d in cursor.description] if cursor.description else []
        return QueryResult(
            ok=True,
            sql=sql,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    except sqlite3.OperationalError as exc:
        msg = str(exc)
        # The progress handler surfaces as "interrupted"; translate it into
        # something the model can act on rather than a cryptic internal word.
        if "interrupted" in msg.lower():
            msg = (
                f"Query exceeded the {timeout:g}s time limit and was cancelled. "
                "It is probably missing a join condition or aggregating over "
                "too many rows -- add the join predicate or narrow the filter."
            )
        else:
            msg = _enrich_error(conn, sql, msg)
        return QueryResult(
            ok=False, sql=sql, error=msg,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    except sqlite3.Error as exc:
        # Covers DatabaseError, IntegrityError, ProgrammingError (e.g. a caller
        # passing multiple statements, which sqlite3.execute refuses).
        return QueryResult(
            ok=False, sql=sql, error=f"{type(exc).__name__}: {exc}",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    except Exception as exc:  # noqa: BLE001 -- the loop must never see a raise
        return QueryResult(
            ok=False, sql=sql, error=f"Unexpected {type(exc).__name__}: {exc}",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    finally:
        _clear_timeout(conn)
        if cursor is not None:
            cursor.close()


def fetch_all(
    sql: str,
    params: Sequence[Any] = (),
    timeout_s: float | None = None,
) -> list[tuple]:
    """Run trusted SQL that this codebase wrote. Raises on failure.

    Used by ``schema.py`` for PRAGMA introspection and by the evaluation
    harness for gold queries. A failure here means a bug in the repo, and
    swallowing it into a result object would hide it.
    """
    conn = get_connection()
    timeout = settings.query_timeout_s if timeout_s is None else timeout_s
    try:
        _install_timeout(conn, timeout)
        return conn.execute(sql, params).fetchall()
    finally:
        _clear_timeout(conn)


def health_check() -> dict[str, Any]:
    """Cheap readiness probe for ``GET /health`` and startup logs."""
    try:
        tables = _list_tables(get_connection())
        result = execute_query("SELECT 1", max_rows=1)
        return {
            "ok": result.ok and bool(tables),
            "db_path": str(settings.db_path),
            "tables": len(tables),
            "read_only": True,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":  # quick manual check: python db.py
    import json

    print(json.dumps(health_check(), indent=2))
    for probe in ("SELECT COUNT(*) FROM orders", "SELECT * FROM nope"):
        r = execute_query(probe)
        print(f"\n{probe}\n  -> {r.preview()}")