"""The agent's two tools: a schema inspector and a query executor.

Your brief asks for two tools with an orchestration loop, and defining them as
real tools rather than inline function calls is what makes this an agent
instead of a pipeline. It is also the part interviewers ask about, so the
distinction is worth being able to defend.

**Each tool exists in two forms, deliberately.**

* ``*_structured`` returns a dataclass. ``nodes.py`` calls these, because the
  graph has to *route* on the outcome -- a guardrail rejection and an empty
  result go to different prompts -- and routing on a parsed string would be
  fragile nonsense.
* ``inspect_schema`` / ``run_sql`` are ``@tool``-decorated and return plain
  text. These are what you bind to a model if you let it choose its own tool
  calls. Their docstrings are part of the prompt: the model reads them to
  decide when to call what, so they are written for the model, not for you.

The structured form is the source of truth; the text form is a thin renderer
over it. Keeping one implementation means the tool the model sees and the tool
the graph runs can never diverge in behaviour.

**The executor never raises.** A failed query returns a value carrying the
database's error text. If it raised, the graph could not route to the
reflection node and the self-correction loop -- the entire point of the
project -- would not exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import prompts, schema
from config import settings
from db import QueryResult, execute_query
from guardrails import validate_sql
from state import FailureKind

try:  # langchain is only needed for the model-facing wrappers
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover
    def tool(fn):  # type: ignore[misc]
        """Fallback so the module imports without langchain installed."""
        return fn


# ---------------------------------------------------------------------------
# Structured results
# ---------------------------------------------------------------------------

@dataclass
class SqlToolResult:
    """Outcome of the executor, in the form the graph routes on."""

    ok: bool
    sql: str
    #: Which failure this was, or None on success. ``nodes.py`` chooses the
    #: retry prompt from this, which is why it is an enum-like field rather
    #: than something to be inferred from the error text.
    kind: FailureKind | None = None
    error: str | None = None
    result: QueryResult | None = None

    @property
    def columns(self) -> list[str] | None:
        return self.result.columns if self.result else None

    @property
    def rows(self) -> list[tuple]:
        return self.result.rows if self.result else []

    @property
    def row_count(self) -> int:
        return self.result.row_count if self.result else 0

    def render(self, max_rows: int = 20) -> str:
        """Text form, for the model or a log."""
        if not self.ok:
            return f"ERROR: {self.error}"
        if self.row_count == 0:
            return "Query executed successfully but returned no rows."
        body = prompts.render_rows_for_prompt(self.columns, self.rows, max_rows)
        note = " (truncated)" if self.result and self.result.truncated else ""
        return f"{self.row_count} row(s){note}\n{body}"


# ---------------------------------------------------------------------------
# Tool 1 -- schema inspector
# ---------------------------------------------------------------------------

def inspect_schema_structured(table_name: str | None = None) -> str:
    """List tables, or describe one table's columns and foreign keys.

    Delegates to ``schema.py`` rather than querying sqlite_master directly, so
    the rendering the model sees here is identical to the schema block in its
    prompt. Two different renderings of the same database is a subtle way to
    confuse a model about what exists.
    """
    if table_name:
        return schema.describe_table(table_name.strip())

    tables = schema.get_tables()
    lines = ["Tables in this database:"]
    for name, table in tables.items():
        cols = ", ".join(table.column_names)
        lines.append(f"  {name} ({cols})")
    return "\n".join(lines)


@tool
def inspect_schema(table_name: str | None = None) -> str:
    """Inspect the database schema.

    Call with no argument to list every table and its columns. Call with a
    table name to see that table's column types, primary key, and the foreign
    keys that connect it to other tables.

    Use this when you are unsure a column exists, or when you need to know how
    two tables relate before writing a JOIN.

    Args:
        table_name: Optional name of a single table to describe in detail.
    """
    return inspect_schema_structured(table_name)


# ---------------------------------------------------------------------------
# Tool 2 -- query executor
# ---------------------------------------------------------------------------

def run_sql_structured(
    sql: str,
    max_rows: int | None = None,
    unlimited: bool = False,
) -> SqlToolResult:
    """Validate and execute a read-only query. Never raises.

    Order matters: guardrails run *before* the database sees anything. That
    keeps an unsafe statement off the connection entirely, and it returns a
    rejection reason phrased as an instruction, so the correction loop treats
    it exactly like a database error and rewrites the query.

    An empty result is reported as ``kind="empty"`` rather than plain success.
    Sometimes zero rows is the right answer and sometimes it means the model
    filtered on a value that does not exist; the graph decides which, and it
    cannot decide if this function collapses both into ``ok=True``.
    """
    check = validate_sql(sql)
    if not check.ok:
        return SqlToolResult(ok=False, sql=sql, kind="guardrail", error=check.reason)

    result = execute_query(sql, max_rows=max_rows, unlimited=unlimited)
    if not result.ok:
        return SqlToolResult(
            ok=False, sql=sql, kind="execution", error=result.error, result=result
        )

    if result.row_count == 0:
        return SqlToolResult(
            ok=True, sql=sql, kind="empty",
            error="Query executed successfully but returned no rows.",
            result=result,
        )

    return SqlToolResult(ok=True, sql=sql, kind=None, result=result)


@tool
def run_sql(sql: str) -> str:
    """Execute a read-only SQL query against the e-commerce database.

    Only a single SELECT statement is allowed. Any attempt to modify data or
    schema will be rejected.

    Returns the resulting rows, or the database's error message if the query
    failed. Read that error carefully and correct the query -- it usually
    names the exact column or table that was wrong.

    Args:
        sql: One SQLite SELECT statement.
    """
    return run_sql_structured(sql).render()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Bind these to a model when you want it to choose its own tool calls.
#: ``graph.py`` does not use this list: its loop calls the structured
#: functions directly, so the control flow is explicit and inspectable rather
#: than left to the model. Both are legitimate agent designs; the explicit
#: graph is easier to evaluate, which is why this project prefers it.
TOOLS = [inspect_schema, run_sql]


def tool_descriptions() -> dict[str, str]:
    """Name -> description, for traces and the README's tool table."""
    out: dict[str, Any] = {}
    for t in TOOLS:
        name = getattr(t, "name", getattr(t, "__name__", str(t)))
        desc = getattr(t, "description", None) or (t.__doc__ or "").strip()
        out[name] = desc.split("\n")[0]
    return out


if __name__ == "__main__":  # python tools.py
    print(inspect_schema_structured())
    print()
    print(inspect_schema_structured("order_items"))
    print()
    for probe in (
        "SELECT order_status, COUNT(*) FROM orders GROUP BY 1",
        "SELECT nope FROM orders",
        "DELETE FROM orders",
        "SELECT * FROM orders WHERE order_status = 'nonexistent'",
    ):
        r = run_sql_structured(probe)
        print(f"{probe}\n  kind={r.kind}  {r.render()[:120]}\n")