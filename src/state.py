"""The typed object passed from node to node in the graph.

LangGraph nodes communicate only through state, so defining it explicitly is
what keeps the loop readable and stops nodes from developing hidden coupling.
Each node receives the whole state and returns a *partial* dict of the keys it
changed; LangGraph merges those into the next state.

Two decisions here matter beyond bookkeeping:

**``history`` uses an append reducer.** Annotating it with ``operator.add``
means a node returns ``{"history": [entry]}`` and LangGraph appends rather than
overwrites. Without the reducer, every node would have to read the existing
list, copy it, append, and return the whole thing -- and the first node that
forgot would silently erase the record of how the agent recovered. That record
is your README's example trace, so it is worth protecting structurally.

**``to_result`` lives here, not in graph.py.** The evaluation harness scores
``baseline.answer`` and ``graph.run_agent`` through one code path, which only
works if both return the same keys. Defining that shape once, next to the
state it is built from, is what stops the two from drifting apart -- and a
drift there would not crash anything, it would just quietly make your
before/after comparison invalid.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from db import QueryResult

#: Why the last attempt failed. Nodes route on this: a guardrail rejection and
#: a database error both go to the correction prompt, but an empty result gets
#: its own prompt, because "no rows" is a different problem from "broken query"
#: and deserves a different question put to the model.
FailureKind = Literal["guardrail", "execution", "empty", "no_sql", "unanswerable"]


class Attempt(TypedDict, total=False):
    """One pass through generate -> validate -> execute.

    Appended to ``history`` whether it succeeded or failed. The failures are
    the interesting part: they are what you show when an interviewer asks how
    the agent recovers.
    """

    n: int
    sql: str | None
    ok: bool
    error: str | None
    kind: FailureKind | None
    row_count: int | None
    elapsed_ms: float | None


class AgentState(TypedDict, total=False):
    """State threaded through the graph.

    ``total=False`` so a node can return only the keys it touched.
    """

    # --- input ---
    question: str

    # --- schema stage ---
    schema_text: str
    #: Tables retrieval selected, and which were added to make them joinable.
    #: Carried purely for tracing; the model only ever sees ``schema_text``.
    retrieved_tables: list[str]
    expanded_tables: list[str]

    # --- generation stage ---
    sql: str | None
    #: The model's unedited response. Kept when no SQL could be extracted, so
    #: a refusal is inspectable instead of vanishing.
    raw_response: str | None

    # --- execution stage ---
    result: QueryResult | None
    error: str | None
    kind: FailureKind | None

    # --- loop control ---
    attempts: int
    history: Annotated[list[Attempt], operator.add]
    done: bool
    #: Set when the agent stops without an answer, e.g. the retry cap was hit.
    give_up_reason: str | None

    # --- output stage ---
    explanation: str | None
    unanswerable: bool


# ---------------------------------------------------------------------------
# Construction and updates
# ---------------------------------------------------------------------------

def new_state(question: str) -> AgentState:
    """Initial state for one question.

    Every key is initialised, including the ones nodes will overwrite. Partial
    state is how ``KeyError`` in a node three steps down the graph happens.
    """
    return AgentState(
        question=question,
        schema_text="",
        retrieved_tables=[],
        expanded_tables=[],
        sql=None,
        raw_response=None,
        result=None,
        error=None,
        kind=None,
        attempts=0,
        history=[],
        done=False,
        give_up_reason=None,
        explanation=None,
        unanswerable=False,
    )


def make_attempt(
    n: int,
    sql: str | None,
    ok: bool,
    error: str | None = None,
    kind: FailureKind | None = None,
    result: QueryResult | None = None,
) -> Attempt:
    """Build one history entry.

    A helper rather than an inline dict literal because every node writes one,
    and consistent keys are what let the tracer and the README trace render
    them uniformly.
    """
    return Attempt(
        n=n,
        sql=sql,
        ok=ok,
        error=error,
        kind=kind,
        row_count=result.row_count if result else None,
        elapsed_ms=round(result.elapsed_ms, 1) if result else None,
    )


def should_retry(state: AgentState, max_retries: int) -> bool:
    """Whether the loop may make another attempt.

    Kept here rather than inline in the conditional edge so the stopping rule
    is stated once and can be unit-tested without building a graph.
    """
    if state.get("done"):
        return False
    if state.get("unanswerable"):
        return False  # a considered refusal, not a failure to retry
    if state.get("error") is None:
        return False
    return state.get("attempts", 0) < max_retries


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

#: The keys every system must return. ``baseline.answer`` and
#: ``graph.run_agent`` are both checked against this, because a mismatch would
#: not raise -- it would silently make the two systems incomparable.
RESULT_KEYS: frozenset[str] = frozenset({
    "system", "question", "sql", "ok", "columns", "rows", "row_count",
    "error", "explanation", "attempts", "history", "unanswerable",
    "raw_response",
})


def to_result(state: AgentState, system: str = "agent") -> dict[str, Any]:
    """Convert final state into the dict the harness and API consume.

    Note ``rows`` comes straight from the ``QueryResult`` and may be truncated
    by ``settings.max_rows`` for the demo UI. The evaluation harness therefore
    re-executes ``sql`` itself with ``unlimited=True`` rather than trusting
    this field -- see ``eval/run_eval.py``.
    """
    result = state.get("result")
    ok = bool(result and result.ok and state.get("error") is None)

    return {
        "system": system,
        "question": state.get("question", ""),
        "sql": state.get("sql"),
        "ok": ok,
        "columns": result.columns if result else None,
        "rows": result.rows if result else [],
        "row_count": result.row_count if result else 0,
        "error": state.get("error") or state.get("give_up_reason"),
        "explanation": state.get("explanation"),
        "attempts": state.get("attempts", 0),
        "history": list(state.get("history", [])),
        "unanswerable": bool(state.get("unanswerable")),
        "raw_response": state.get("raw_response"),
        # Extras beyond the shared contract: ignored by the harness, useful in
        # the UI and in traces.
        "retrieved_tables": list(state.get("retrieved_tables", [])),
        "elapsed_ms": round(result.elapsed_ms, 1) if result else 0.0,
    }


def trace_summary(state: AgentState) -> str:
    """One-line-per-attempt rendering of how the agent got there.

    This is the "example run trace" your README needs. A question that failed
    once and recovered is the single most persuasive artifact in the project,
    so make it easy to paste.
    """
    lines = [f"Q: {state.get('question', '')}"]
    tables = state.get("retrieved_tables")
    if tables:
        lines.append(f"schema: {', '.join(tables)}")
    for a in state.get("history", []):
        status = "ok" if a.get("ok") else f"FAILED ({a.get('kind')})"
        sql = " ".join((a.get("sql") or "").split())[:100]
        lines.append(f"  attempt {a.get('n')}: {status}")
        if sql:
            lines.append(f"    sql: {sql}")
        if a.get("error"):
            lines.append(f"    err: {str(a['error'])[:120]}")
    if state.get("explanation"):
        lines.append(f"answer: {state['explanation']}")
    if state.get("give_up_reason"):
        lines.append(f"gave up: {state['give_up_reason']}")
    return "\n".join(lines)