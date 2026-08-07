"""Wires the nodes together and owns the retry edge. The public entry point.

Everything else in the library exists to be called from here, and this file
deliberately contains no logic of its own -- the node bodies live in
``nodes.py`` and the routing rules live there too, so this stays a readable
diagram of control flow rather than a place bugs hide.

The graph::

    START
      |
      v
    retrieve_schema
      |
      v
    generate_sql ---------[route_after_generate]---> give_up
      |                                                 ^
      v                                                 |
    validate -------------[route_after_validate]--------+
      |                          |                      |
      | execute                  | reflect (guardrail)  |
      v                          v                      |
    execute_sql            reflect ---[route_after_generate]
      |                          ^
      [route_after_execute]      |
      |        |                 |
      | explain| reflect --------+
      v        |
    explain -> END           give_up -> END

Both ``graph.run_agent`` and ``baseline.answer`` return the same dict shape, so
``eval/run_eval.py`` scores them through one code path. That is what
guarantees the system you measure is the system you deploy.
"""

from __future__ import annotations

import argparse
import logging
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

import nodes
from config import settings
from state import AgentState, new_state, to_result, trace_summary

logger = logging.getLogger(__name__)

#: Ceiling on total node visits. With ``max_retries`` at 3 the loop runs at
#: most a dozen or so nodes, so this is a backstop against a routing bug
#: rather than a tuning knob -- if you ever hit it, a router is wrong.
RECURSION_LIMIT = 50


def build_graph():
    """Assemble and compile the agent graph.

    Note the two ``reflect`` entry points. A guardrail rejection reaches it
    from ``validate`` without ever touching the database, and an execution
    failure or empty result reaches it from ``execute_sql``. Both then rejoin
    the same path -- ``reflect`` produces new SQL, so it routes through
    ``route_after_generate`` exactly like the first attempt did.
    """
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_schema", nodes.retrieve_schema)
    graph.add_node("generate_sql", nodes.generate_sql)
    graph.add_node("validate", nodes.validate_sql)
    graph.add_node("execute", nodes.execute_sql)
    graph.add_node("reflect", nodes.reflect)
    graph.add_node("explain", nodes.explain)
    graph.add_node("give_up", nodes.give_up)

    graph.add_edge(START, "retrieve_schema")
    graph.add_edge("retrieve_schema", "generate_sql")

    # A refusal or a model outage must not march on into validation holding
    # no SQL -- it would report a confusing downstream error instead of the
    # real one.
    graph.add_conditional_edges(
        "generate_sql", nodes.route_after_generate,
        {"validate": "validate", "give_up": "give_up"},
    )

    # A rejected statement skips the database entirely.
    graph.add_conditional_edges(
        "validate", nodes.route_after_validate,
        {"execute": "execute", "reflect": "reflect", "give_up": "give_up"},
    )

    # The retry edge: this is the project.
    graph.add_conditional_edges(
        "execute", nodes.route_after_execute,
        {"explain": "explain", "reflect": "reflect", "give_up": "give_up"},
    )

    # Corrected SQL re-enters the same validate -> execute path.
    graph.add_conditional_edges(
        "reflect", nodes.route_after_generate,
        {"validate": "validate", "give_up": "give_up"},
    )

    graph.add_edge("explain", END)
    graph.add_edge("give_up", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_graph():
    """Compiled graph, built once. Compilation is not free per question."""
    return build_graph()


def _callbacks() -> list[Any]:
    """Tracing callbacks, if tracing is configured.

    Optional and fail-soft: a missing Langfuse key, or ``tracing.py`` not
    existing yet, must not break a local run or the deployed demo.
    """
    try:
        from tracing import get_callback_handler
        handler = get_callback_handler(settings.run_label)
        return [handler] if handler else []
    except Exception as exc:  # noqa: BLE001
        logger.debug("tracing unavailable: %s", exc)
        return []


def run_agent(question: str) -> dict[str, Any]:
    """Answer one question. The single public entry point. Never raises.

    Returns the same dict shape as ``baseline.answer`` -- see
    ``state.RESULT_KEYS``. A failure anywhere comes back as ``ok=False`` with
    a reason, because a fifty-question evaluation run must not die on
    question seventeen.
    """
    initial = new_state(question)

    try:
        final = get_graph().invoke(
            initial,
            config={"recursion_limit": RECURSION_LIMIT, "callbacks": _callbacks()},
        )
    except Exception as exc:  # noqa: BLE001
        # Includes GraphRecursionError. Reaching here means a routing bug,
        # not a bad query -- surfaced as a result so the harness records it
        # against the question instead of losing the whole run.
        logger.exception("graph execution failed")
        failed = dict(initial)
        failed.update(done=True, give_up_reason=f"{type(exc).__name__}: {exc}")
        return to_result(failed, system="agent")

    return to_result(final, system="agent")


def render_diagram() -> str:
    """Mermaid source for the graph.

    Your README needs an architecture diagram, and one generated from the
    actual compiled graph cannot drift out of date the way a hand-drawn one
    does. Paste the output into a ```mermaid block.
    """
    try:
        return get_graph().get_graph().draw_mermaid()
    except Exception as exc:  # noqa: BLE001
        return f"(diagram unavailable: {exc})"


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Ask the agent one question.")
    parser.add_argument("question", nargs="?", help="Natural-language question")
    parser.add_argument("--trace", action="store_true", help="Print the run trace.")
    parser.add_argument("--diagram", action="store_true", help="Print Mermaid diagram.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.diagram:
        print(render_diagram())
        return
    if not args.question:
        parser.error("a question is required (or use --diagram)")
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    out = run_agent(args.question)

    print(f"\nSQL:\n{out['sql']}\n")
    if out["ok"]:
        if out["columns"]:
            print(" | ".join(out["columns"]))
        for row in out["rows"][:10]:
            print(" | ".join("NULL" if v is None else str(v) for v in row))
        if out["row_count"] > 10:
            print(f"... and {out['row_count'] - 10} more")
        if out["explanation"]:
            print(f"\n{out['explanation']}")
    else:
        print(f"FAILED: {out['error']}")

    print(f"\nattempts: {out['attempts']}")
    if args.trace:
        print()
        # Rebuild a state-like dict for the trace renderer.
        print(trace_summary({
            "question": out["question"],
            "retrieved_tables": out.get("retrieved_tables", []),
            "history": out["history"],
            "explanation": out["explanation"],
            "give_up_reason": None if out["ok"] else out["error"],
        }))

    from llm import usage_summary
    print(f"\nusage: {usage_summary()}")


if __name__ == "__main__":  # python graph.py "how many orders?"
    _cli()