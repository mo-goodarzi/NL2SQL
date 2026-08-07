"""Turn raw rows into something a person can read.

The demo is judged on legibility, and separating presentation from agent logic
means the API can return structured data while the UI renders it however it
likes. Nothing here changes what the agent computed -- these functions are
display only, and are never used by ``eval/compare.py``. That separation is
deliberate: formatting rounds and truncates, and a formatter in the comparison
path would silently change what counts as a correct answer.

The one function here that calls a model is ``explain_result``. It lives in
this module rather than in ``nodes.py`` because turning results into prose is
presentation, and ``nodes.explain`` delegates to it -- one implementation, so
the CLI, the API, and the graph cannot drift into explaining things
differently.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import prompts
from db import QueryResult
from llm import LLMError, complete

logger = logging.getLogger(__name__)

#: Values longer than this are elided in tables. Review comments in this
#: dataset run to paragraphs and would destroy any column alignment.
MAX_CELL_CHARS = 60


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

def format_value(value: Any, max_chars: int = MAX_CELL_CHARS) -> str:
    """Render one cell.

    Floats get thousand separators and two decimals, which is right for the
    money and averages this dataset is full of. Integers keep separators but
    no decimals, so a count of 99441 does not read as 99441.00.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value != value:  # NaN
            return "NaN"
        if value.is_integer() and abs(value) < 1e15:
            return f"{int(value):,}"
        return f"{value:,.2f}"

    text = str(value)
    if len(text) > max_chars:
        return text[: max_chars - 1] + "\u2026"
    return text


def _is_all_null(rows: Sequence[Sequence[Any]]) -> bool:
    """One row, every column NULL.

    Worth detecting because it is not the same as an empty result and is easy
    to misread. An aggregate over zero matching rows returns exactly this --
    ``SELECT SUM(price) ... WHERE category = 'health_beauty'`` gives one row
    containing NULL, not zero rows -- so a bare "NULL" on screen looks like a
    data problem when it actually means the filter matched nothing.
    """
    return len(rows) == 1 and all(v is None for v in rows[0])


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def rows_to_markdown(
    columns: Sequence[str] | None,
    rows: Sequence[Sequence[Any]],
    max_rows: int = 20,
) -> str:
    """Markdown table, for Streamlit and the README."""
    if not columns:
        return "_(no columns)_"
    if not rows:
        return "_(no rows)_"

    def esc(text: str) -> str:
        # An unescaped pipe inside a value silently breaks the table.
        return text.replace("|", "\\|")

    head = "| " + " | ".join(esc(str(c)) for c in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(esc(format_value(v)) for v in row) + " |"
        for row in rows[:max_rows]
    ]
    out = "\n".join([head, rule, *body])
    if len(rows) > max_rows:
        out += f"\n\n_\u2026 and {len(rows) - max_rows:,} more row(s)_"
    return out


def rows_to_text(
    columns: Sequence[str] | None,
    rows: Sequence[Sequence[Any]],
    max_rows: int = 20,
) -> str:
    """Column-aligned plain text, for the CLI and logs."""
    if not columns:
        return "(no columns)"
    if not rows:
        return "(no rows)"

    shown = [[format_value(v) for v in row] for row in rows[:max_rows]]
    widths = [
        max(len(str(col)), *(len(r[i]) for r in shown)) if shown else len(str(col))
        for i, col in enumerate(columns)
    ]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    out = [line([str(c) for c in columns]), line(["-" * w for w in widths])]
    out.extend(line(r) for r in shown)
    if len(rows) > max_rows:
        out.append(f"... and {len(rows) - max_rows:,} more row(s)")
    return "\n".join(out)


def summarise_result(result: QueryResult | None) -> str:
    """One line describing the shape of a result, for headers and logs."""
    if result is None:
        return "no result"
    if not result.ok:
        return f"failed: {result.error}"
    if result.row_count == 0:
        return "0 rows"
    if _is_all_null(result.rows):
        return "1 row (NULL \u2014 no rows matched the filter)"
    suffix = " (truncated)" if result.truncated else ""
    cols = len(result.columns or [])
    return f"{result.row_count:,} row(s) \u00d7 {cols} column(s){suffix}"


def is_scalar(result: QueryResult | None) -> bool:
    """A single cell -- ``SELECT COUNT(*)`` and friends.

    Worth special-casing: a one-by-one markdown table around the number 60 is
    worse than just writing 60.
    """
    return bool(
        result and result.ok and result.row_count == 1
        and result.columns and len(result.columns) == 1
    )


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------

def explain_result(
    question: str,
    result: QueryResult | None,
    max_rows: int = 20,
) -> str | None:
    """One-sentence natural-language answer. Returns None if unavailable.

    Fail-soft by design: a missing explanation must never turn an otherwise
    correct answer into a failure. The SQL and the rows are the substance;
    this is a convenience on top.
    """
    if result is None or not result.ok:
        return None

    if result.row_count == 0:
        # Answered locally rather than spending a model call to say "nothing".
        return "No rows matched that query."
    if _is_all_null(result.rows):
        return ("The query returned no value, which means no rows matched the "
                "filter conditions.")

    try:
        text = complete(
            system=prompts.EXPLANATION_SYSTEM,
            user=prompts.build_explanation_prompt(
                question, result.columns, result.rows, max_rows
            ),
            max_tokens=200,
        )
    except LLMError as exc:
        logger.warning("explanation failed: %s", exc)
        return None
    return text.strip() or None


# ---------------------------------------------------------------------------
# Whole-answer assembly
# ---------------------------------------------------------------------------

def format_answer(
    result: dict[str, Any],
    style: str = "text",
    max_rows: int = 20,
    show_sql: bool = True,
) -> str:
    """Render a full ``run_agent`` / ``baseline.answer`` result.

    ``show_sql`` defaults to True and should stay that way in the demo. Visible
    SQL is what makes the agent legible instead of a black box, and it is the
    first thing a reviewer looks for.
    """
    md = style == "markdown"
    parts: list[str] = []

    if result.get("unanswerable"):
        return "This question cannot be answered from the available data."

    if show_sql and result.get("sql"):
        parts.append(
            f"```sql\n{result['sql']}\n```" if md else f"SQL:\n{result['sql']}"
        )

    if not result.get("ok"):
        error = result.get("error") or "the query could not be completed"
        attempts = result.get("attempts", 0)
        note = f" after {attempts} attempt(s)" if attempts else ""
        parts.append(f"**Failed{note}:** {error}" if md else f"FAILED{note}: {error}")
        return "\n\n".join(parts)

    if result.get("explanation"):
        parts.append(result["explanation"])

    columns, rows = result.get("columns"), result.get("rows") or []
    if rows and _is_all_null(rows):
        parts.append("No rows matched the filter conditions.")
    elif not rows:
        parts.append("No rows matched that query.")
    elif len(rows) == 1 and columns and len(columns) == 1:
        parts.append(f"**{format_value(rows[0][0])}**" if md
                     else format_value(rows[0][0]))
    else:
        parts.append(rows_to_markdown(columns, rows, max_rows) if md
                     else rows_to_text(columns, rows, max_rows))

    if result.get("attempts", 0) > 1:
        # Surfacing the retry is a feature: it shows self-correction happened.
        n = result["attempts"]
        parts.append(f"_Answered after {n} attempts (self-corrected)._" if md
                     else f"(answered after {n} attempts \u2014 self-corrected)")

    return "\n\n".join(parts)