"""One plain function per step of the loop.

Keeping node logic out of ``graph.py`` means that file stays a readable
diagram of control flow, and every node here can be called with a dict and
asserted on without constructing a graph at all.

Each node takes the whole state and returns **only the keys it changed**.
LangGraph merges the partial dict; ``history`` is annotated with an append
reducer in ``state.py``, so returning ``{"history": [entry]}`` appends rather
than overwrites.

Two structural points worth knowing before reading the code:

**``reflect`` is a generation step, not a bookkeeping step.** It calls the
model with a different prompt -- the failed SQL plus the database's own error
-- and produces corrected SQL directly. So the loop is
``generate -> validate -> execute -> reflect -> validate -> execute``, and
reflect is simply generate wearing a different prompt. Treating it as its own
node keeps the retry visible in the graph diagram and in traces, which is the
thing you want an interviewer to see.

**No node raises.** A model outage, a guardrail rejection, and a bad column
all end as state, because a raise anywhere in here would unwind the graph and
destroy the ``history`` that documents how the agent recovered.
"""

from __future__ import annotations

import logging
from typing import Any

import prompts, schema
from config import settings
from llm import LLMError, complete, extract_sql
from state import AgentState, make_attempt, should_retry
from tools import run_sql_structured

logger = logging.getLogger(__name__)


def _is_unanswerable(raw: str) -> bool:
    """The escape hatch offered by ``SQL_SYSTEM``.

    Checked against the raw response, before ``extract_sql`` runs: extraction
    looks for SELECT/WITH and would strip this to an empty string, turning a
    deliberate refusal into a "no SQL returned" failure. The two mean quite
    different things and your report should count them apart.
    """
    return raw.strip().upper().startswith("UNANSWERABLE")


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------

def retrieve_schema(state: AgentState) -> dict[str, Any]:
    """Put the schema the model will see into state.

    Routes through ``schema.get_schema_text``, the single place that honours
    ``settings.use_schema_retrieval`` -- so flipping the Step 4 flag changes
    behaviour without touching the graph. The selected table names are also
    recorded, purely so a trace can show *why* a query failed when retrieval
    left something out.
    """
    question = state["question"]

    if settings.use_schema_retrieval:
        retrieval = schema.retrieve_tables(question)
        tables = {name: t for name, t in schema.get_tables().items()}
        schema_text = schema.render_schema(
            [tables[n] for n in retrieval.tables if n in tables],
            include_samples=settings.include_sample_values,
        )
        return {
            "schema_text": schema_text,
            "retrieved_tables": retrieval.tables,
            "expanded_tables": retrieval.added_by_expansion,
        }

    return {
        "schema_text": schema.get_full_schema_text(),
        "retrieved_tables": schema.get_table_names(),
        "expanded_tables": [],
    }


# ---------------------------------------------------------------------------
# 2. Generate
# ---------------------------------------------------------------------------

def generate_sql(state: AgentState) -> dict[str, Any]:
    """First attempt: turn the question into SQL."""
    try:
        raw = complete(
            system=prompts.SQL_SYSTEM,
            user=prompts.build_sql_prompt(state["question"], state["schema_text"]),
        )
    except LLMError as exc:
        # llm.py already exhausted transport retries; retrying here would only
        # burn the semantic budget on an outage that has nothing to do with SQL.
        return {
            "error": f"Model call failed: {exc}",
            "kind": "no_sql",
            "done": True,
            "give_up_reason": f"Model call failed: {exc}",
        }

    if _is_unanswerable(raw):
        return {
            "unanswerable": True,
            "done": True,
            "raw_response": raw.strip()[:500],
            "error": "Model judged the question unanswerable from this schema.",
            "kind": "unanswerable",
        }

    sql = extract_sql(raw)
    if not sql:
        return {
            "raw_response": raw.strip()[:500],
            "error": "Model returned no SQL.",
            "kind": "no_sql",
            "done": True,
        }

    return {"sql": sql, "error": None, "kind": None}


# ---------------------------------------------------------------------------
# 3. Validate
# ---------------------------------------------------------------------------

def validate_sql(state: AgentState) -> dict[str, Any]:
    """Guardrail check before the query reaches the database.

    ``run_sql_structured`` validates again internally, which is deliberate
    redundancy rather than an oversight: validation is a microsecond string
    scan, having it as its own node keeps the graph diagram honest about what
    happens, and keeping it inside the tool means the tool stays safe if
    anything ever calls it directly.

    A rejection is recorded exactly like a database error -- same fields, same
    downstream routing -- so the agent learns to rewrite unsafe queries
    instead of merely being refused.
    """
    from guardrails import validate_sql as check

    sql = state.get("sql")
    if not sql:
        return {"error": "No SQL to validate.", "kind": "no_sql", "done": True}

    result = check(sql)
    if result.ok:
        return {"error": None, "kind": None}

    attempt_n = state.get("attempts", 0) + 1
    return {
        "error": result.reason,
        "kind": "guardrail",
        "attempts": attempt_n,
        "history": [make_attempt(attempt_n, sql, ok=False,
                                 error=result.reason, kind="guardrail")],
    }


# ---------------------------------------------------------------------------
# 4. Execute
# ---------------------------------------------------------------------------

def execute_sql(state: AgentState) -> dict[str, Any]:
    """Run the query and record the attempt.

    ``max_rows`` applies here because this result may be rendered in the demo
    UI. The evaluation harness re-executes the SQL itself with
    ``unlimited=True`` rather than trusting these rows, so truncation cannot
    corrupt a comparison.
    """
    sql = state.get("sql")
    if not sql:
        return {"error": "No SQL to execute.", "kind": "no_sql", "done": True}

    tool_result = run_sql_structured(sql)
    attempt_n = state.get("attempts", 0) + 1

    entry = make_attempt(
        attempt_n, sql,
        ok=tool_result.ok and tool_result.kind is None,
        error=tool_result.error,
        kind=tool_result.kind,
        result=tool_result.result,
    )

    return {
        "result": tool_result.result,
        # An empty result carries kind="empty" while ok is True, so the router
        # can send it to a prompt that asks the right question about it.
        "error": tool_result.error if tool_result.kind else None,
        "kind": tool_result.kind,
        "attempts": attempt_n,
        "history": [entry],
    }


# ---------------------------------------------------------------------------
# 5. Reflect -- the heart of the project
# ---------------------------------------------------------------------------

def reflect(state: AgentState) -> dict[str, Any]:
    """Read the failure and write a corrected query.

    The prompt is chosen from ``kind``: an empty result gets its own prompt
    because "no rows matched" is a different problem from "the query broke",
    and asking the model the right question is the difference between a
    productive retry and a repeat of the same mistake.

    The prior attempts are passed along too. Without them a model will happily
    resubmit a query it has already tried, spending the retry budget on an
    attempt that is guaranteed to fail.
    """
    question = state["question"]
    schema_text = state["schema_text"]
    failed_sql = state.get("sql") or ""
    error = state.get("error") or "unknown error"
    kind = state.get("kind")

    if kind == "empty":
        user = prompts.build_empty_result_prompt(question, failed_sql, schema_text)
    else:
        user = prompts.build_correction_prompt(
            question, failed_sql, error, schema_text,
            history=state.get("history", []),
        )

    try:
        raw = complete(system=prompts.SQL_SYSTEM, user=user)
    except LLMError as exc:
        return {
            "done": True,
            "give_up_reason": f"Model call failed during correction: {exc}",
        }

    if _is_unanswerable(raw):
        return {
            "unanswerable": True,
            "done": True,
            "raw_response": raw.strip()[:500],
            "error": "Model judged the question unanswerable after retrying.",
            "kind": "unanswerable",
        }

    corrected = extract_sql(raw)
    if not corrected:
        return {
            "raw_response": raw.strip()[:500],
            "error": "Model returned no SQL when asked to correct.",
            "kind": "no_sql",
            "done": True,
        }

    if corrected.strip() == failed_sql.strip():
        # It reproduced the failing query. Retrying again would loop on the
        # same error, so stop and say so plainly rather than burning budget.
        return {
            "sql": corrected,
            "done": True,
            "give_up_reason": "Model repeated the same failing query.",
        }

    logger.debug("reflect: rewrote query after %s failure", kind)
    return {"sql": corrected, "error": None, "kind": None}


# ---------------------------------------------------------------------------
# 6. Explain
# ---------------------------------------------------------------------------

def explain(state: AgentState) -> dict[str, Any]:
    """Turn a successful result into one plain sentence.

    Delegates to ``formatter.explain_result`` rather than building the prompt
    here, so the graph, the CLI, and the API all explain results identically.

    Costs a second model call per question. That is worth it in the demo and
    wasted in evaluation, where only the SQL and the rows are scored -- set
    ``EXPLAIN_ANSWERS=false`` for eval runs and halve your cost per sweep.
    """
    if not getattr(settings, "explain_answers", True):
        return {"done": True}

    from formatter import explain_result

    text = explain_result(state["question"], state.get("result"))
    return {"explanation": text, "done": True} if text else {"done": True}


def give_up(state: AgentState) -> dict[str, Any]:
    """Terminal node when the retry budget is exhausted.

    Always ends cleanly with a readable message. An agent that loops forever
    on an impossible question is a bad demo and a worse evaluation run.
    """
    attempts = state.get("attempts", 0)
    reason = state.get("give_up_reason") or (
        f"Gave up after {attempts} attempt(s). Last error: "
        f"{state.get('error') or 'unknown'}"
    )
    return {"done": True, "give_up_reason": reason}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_generate(state: AgentState) -> str:
    """Decide where to go after SQL generation.

    Without this edge the graph would march into validation holding no SQL
    after a refusal or a model outage, and report a confusing downstream error
    instead of the real one. Returns ``"validate"`` or ``"give_up"``.
    """
    if state.get("done") or not state.get("sql"):
        return "give_up"
    return "validate"


def route_after_validate(state: AgentState) -> str:
    """Decide where to go after the guardrail check.

    A rejection skips execution entirely -- there is no point sending a
    statement to the database that we have already decided not to run -- and
    goes straight to correction, because the rejection reason is phrased as an
    instruction the model can act on.

    Returns ``"execute"``, ``"reflect"``, or ``"give_up"``.
    """
    if state.get("done"):
        return "give_up"
    if state.get("kind") != "guardrail":
        return "execute"
    if not settings.use_self_correction:
        return "give_up"
    return "reflect" if should_retry(state, settings.max_retries) else "give_up"


def route_after_execute(state: AgentState) -> str:
    """Decide where to go after an execution attempt.

    Lives here rather than inline in ``graph.py`` so the branch can be tested
    with a plain dict. ``graph.py`` only wires it up.

    Returns one of: ``"explain"``, ``"reflect"``, ``"give_up"``.
    """
    if state.get("unanswerable") or state.get("done"):
        return "give_up" if state.get("error") else "explain"

    if state.get("kind") is None and state.get("error") is None:
        return "explain"

    if not settings.use_self_correction:
        # Step 4 ablation: with correction off, one failure is terminal.
        return "give_up"

    return "reflect" if should_retry(state, settings.max_retries) else "give_up"