"""The demo a recruiter will click.

    streamlit run app/ui.py

This is the artifact that gets looked at first and for about thirty seconds,
so it is built around one idea: **show the reasoning, not just the answer.**
The generated SQL is always visible, and when the agent corrects itself the
failed attempt and the database error that prompted the fix are shown too.
An agent that silently returns a number is indistinguishable from a lookup
table; one that visibly recovers from its own mistake is obviously an agent.

**Two ways to reach the agent.** If ``API_URL`` is set, the UI calls the
FastAPI service over HTTP; otherwise it imports ``run_agent`` and runs
in-process. That is not indecision -- Hugging Face Spaces runs one process per
container, so in-process is what actually deploys, while the HTTP path proves
the API is real and lets you run the two separately on a host that supports
it. Same agent either way.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

#: Set this to call a separately deployed API instead of running in-process.
API_URL = os.getenv("API_URL", "").rstrip("/")

REPO_URL = os.getenv("REPO_URL", "https://github.com/mo-goodarzi/NL2SQL")

EXAMPLES = [
    "How many orders are in the database?",
    "What are the top 5 states by number of customers?",
    # Deliberately phrased in English against a Portuguese column: this one
    # usually returns zero rows first, and the agent recovers by joining the
    # translation table. It is the best thing in the demo -- keep it.
    "Which products are in the health_beauty category?",
    "What is the total revenue for each product category?",
    "What is the average review score by month in 2017?",
]

st.set_page_config(
    page_title="Text-to-SQL Analytics Agent",
    page_icon="\U0001F50D",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

def ask(question: str) -> dict[str, Any]:
    """Send a question to the agent, over HTTP or in-process."""
    if API_URL:
        import requests

        response = requests.post(
            f"{API_URL}/query", json={"question": question}, timeout=120
        )
        response.raise_for_status()
        return response.json()

    from graph import run_agent

    result = run_agent(question)
    result["rows"] = [list(r) for r in result["rows"]]
    result["self_corrected"] = result["attempts"] > 1
    return result


@st.cache_data(ttl=300)
def load_schema() -> list[dict[str, Any]]:
    """Table list for the sidebar. Cached -- the schema does not change."""
    if API_URL:
        import requests

        return requests.get(f"{API_URL}/schema", timeout=30).json()["tables"]

    import schema

    return [
        {"name": name, "columns": [{"name": c.name, "type": c.type}
                                   for c in table.columns]}
        for name, table in schema.get_tables().items()
    ]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_correction(result: dict[str, Any]) -> None:
    """Show the failed attempts that preceded a successful answer.

    Expanded by default when a correction happened. This is the single most
    persuasive element on the page and burying it behind a click wastes it.
    """
    history = result.get("history") or []
    failed = [h for h in history if not h.get("ok")]
    if not failed:
        return

    st.warning(
        f"The first query failed. The agent read the database error and "
        f"rewrote it \u2014 answered on attempt {result['attempts']}."
    )
    with st.expander(f"See what went wrong ({len(failed)} failed attempt(s))",
                     expanded=True):
        for attempt in failed:
            st.caption(f"Attempt {attempt.get('n')} \u2014 {attempt.get('kind')}")
            if attempt.get("sql"):
                st.code(attempt["sql"], language="sql")
            if attempt.get("error"):
                st.error(attempt["error"])


def render_result(result: dict[str, Any]) -> None:
    if result.get("unanswerable"):
        st.info("This question cannot be answered from the available data.")
        return

    # SQL first, always. It is the point of the demo.
    if result.get("sql"):
        st.subheader("Generated SQL")
        st.code(result["sql"], language="sql")

    render_correction(result)

    if not result.get("ok"):
        st.error(result.get("error") or "The query could not be completed.")
        return

    if result.get("explanation"):
        st.subheader("Answer")
        st.markdown(f"### {result['explanation']}")

    rows, columns = result.get("rows") or [], result.get("columns") or []
    st.subheader("Results")

    if not rows:
        st.info("No rows matched that query.")
    elif len(rows) == 1 and all(v is None for v in rows[0]):
        # An aggregate over zero matching rows returns one NULL, not no rows.
        # Unexplained, a bare "None" in a table looks like a bug.
        st.info("The query returned no value \u2014 no rows matched the filters.")
    elif len(rows) == 1 and len(columns) == 1:
        st.metric(columns[0], f"{rows[0][0]:,}" if isinstance(rows[0][0], (int, float))
                  else str(rows[0][0]))
    else:
        st.dataframe(pd.DataFrame(rows, columns=columns),
                     use_container_width=True, hide_index=True)

    cols = st.columns(4)
    cols[0].metric("Rows", f"{result.get('row_count', 0):,}")
    cols[1].metric("Attempts", result.get("attempts", 0))
    cols[2].metric("Self-corrected", "Yes" if result.get("self_corrected") else "No")
    if result.get("elapsed_ms"):
        cols[3].metric("Time", f"{result['elapsed_ms'] / 1000:.1f}s")

    if result.get("retrieved_tables"):
        st.caption("Tables used: " + ", ".join(result["retrieved_tables"]))
    if result.get("trace_url"):
        st.caption(f"[View trace]({result['trace_url']})")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Text-to-SQL Analytics Agent")
st.caption(
    "Ask a question in plain English. The agent inspects the schema, writes "
    "SQL, runs it read-only, and corrects itself if the query fails."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        "A LangGraph agent over the Olist Brazilian e-commerce dataset "
        "(~100k orders across 9 tables).\n\n"
        "- Writes and executes its own SQL\n"
        "- Retries after reading the database error\n"
        "- Read-only: `SELECT` statements only"
    )
    st.markdown(f"[Source on GitHub]({REPO_URL})")

    st.divider()
    st.subheader("Database")
    try:
        for table in load_schema():
            with st.expander(table["name"]):
                st.caption(", ".join(c["name"] for c in table["columns"]))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load schema: {exc}")

    st.divider()
    st.caption(f"Backend: {'API at ' + API_URL if API_URL else 'in-process'}")

if "question" not in st.session_state:
    st.session_state.question = ""

st.subheader("Try an example")
# Buttons write into session state and rerun; Streamlit re-executes the whole
# script on interaction, so the text input picks the value up on the next pass.
for row_start in (0, 3):
    for col, example in zip(st.columns(3), EXAMPLES[row_start:row_start + 3]):
        if col.button(example, use_container_width=True, key=f"ex_{example}"):
            st.session_state.question = example
            st.rerun()

question = st.text_input(
    "Your question",
    value=st.session_state.question,
    placeholder="e.g. Which sellers had the highest average review score?",
) or ""

submitted = st.button("Ask", type="primary")

if submitted and not question.strip():
    st.warning("Please enter a question.")
elif question.strip() and (submitted or question != st.session_state.get("last_asked")):
    st.session_state.last_asked = question
    with st.spinner("Inspecting schema, writing SQL, running it\u2026"):
        try:
            result = ask(question.strip())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not reach the agent: {exc}")
            result = None
    if result:
        st.divider()
        render_result(result)
