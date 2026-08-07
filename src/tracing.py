"""Instrumentation so every run is inspectable.

Tracing is what turns debugging from guesswork into reading. It is also where
your README's example trace comes from -- a question that failed once and
recovered is the most persuasive artifact in the project, and a screenshot of
it beats any amount of prose.

**Everything here fails soft.** A missing Langfuse key, an unreachable host, or
the package not being installed must never break a local run or the deployed
demo. Observability that can take down the thing it observes is worse than no
observability, and a demo that 500s because a tracing key expired is a bad
look in exactly the moment you cannot afford one.

**What gets traced, and the gap you should know about.** ``graph.py`` attaches
the callback handler, so LangGraph reports the node structure --
retrieve_schema, generate_sql, validate, execute, reflect, explain -- and a
trace reads like the loop diagram. But ``llm.py`` calls the provider SDK
directly rather than through LangChain, so the model calls themselves are
*not* captured by that handler. Use ``generation_span`` to close that gap; the
one-line hook needed in ``llm.complete`` is documented on that function.

Written against the Langfuse v4 SDK, whose API differs from v2 and v3. The
imports are guarded so an older version degrades to "tracing off" rather than
crashing on import.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
from functools import lru_cache
from typing import Any, Iterator

from config import settings

logger = logging.getLogger(__name__)

_warned = False


def _warn_once(message: str) -> None:
    """Complain about tracing exactly once per process.

    A warning per question would bury the output of a 42-question evaluation
    run in noise about an optional feature.
    """
    global _warned
    if not _warned:
        logger.warning("tracing disabled: %s", message)
        _warned = True


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_langfuse() -> Any | None:
    """Configured Langfuse client, or None if tracing is not available.

    Cached because constructing a client opens a background flush thread; one
    per process is the intent.
    """
    if not settings.tracing_enabled:
        return None

    try:
        from langfuse import Langfuse
    except ImportError:
        _warn_once("langfuse is not installed (pip install langfuse)")
        return None

    try:
        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            environment=os.getenv("ENVIRONMENT", "development"),
        )
    except Exception as exc:  # noqa: BLE001
        _warn_once(f"could not create client: {exc}")
        return None

    # Register the flush here rather than asking every caller to remember.
    # Langfuse batches in the background, so a short-lived script -- which is
    # exactly what run_eval.py and the CLI are -- can exit with a full buffer
    # and silently lose the traces it just produced.
    atexit.register(flush)
    return client


def flush() -> None:
    """Send buffered traces. Safe to call when tracing is off.

    Call this at the end of any short-lived process. It is registered with
    ``atexit`` automatically, but an explicit call before printing a report
    means the trace link is live by the time someone clicks it.
    """
    client = get_langfuse()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:  # noqa: BLE001
        logger.debug("flush failed: %s", exc)


def check_connection() -> bool:
    """Verify credentials actually work. Useful on startup, not per request."""
    client = get_langfuse()
    if client is None:
        return False
    try:
        return bool(client.auth_check())
    except Exception as exc:  # noqa: BLE001
        _warn_once(f"auth check failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def trace_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything needed to reproduce a run, attached to its trace.

    The same fields evaluation reports embed. Having them on the trace too
    means that when you find an interesting failure in the dashboard weeks
    later, you can tell which configuration produced it without cross-
    referencing anything.
    """
    import prompts

    meta: dict[str, Any] = {
        "run_label": settings.run_label,
        **settings.active_flags,
        **prompts.prompt_metadata(),
    }
    if extra:
        meta.update(extra)
    return meta


def trace_tags() -> list[str]:
    """Short tags for filtering runs in the dashboard.

    Encoding the active flags means you can pull up "every run with schema
    retrieval on" in one click, which is how you find out whether a Step 4
    improvement helped or merely moved failures around.
    """
    tags = [settings.run_label, f"model:{settings.model}"]
    if settings.use_schema_retrieval:
        tags.append("schema_retrieval")
    if settings.use_few_shot:
        tags.append("few_shot")
    if not settings.use_self_correction:
        tags.append("no_self_correction")
    return tags


# ---------------------------------------------------------------------------
# LangChain / LangGraph callback
# ---------------------------------------------------------------------------

def get_callback_handler(run_label: str | None = None) -> Any | None:
    """Callback handler for ``graph.py`` to attach, or None.

    Returning None rather than raising is the contract ``graph.py`` relies on:
    it treats an empty callback list as "tracing off" and carries on.
    """
    if get_langfuse() is None:
        return None

    try:
        from langfuse.langchain import CallbackHandler
    except ImportError as exc:
        # The langchain extra is a separate install from the core SDK.
        _warn_once(f"langfuse langchain integration unavailable: {exc}")
        return None

    try:
        return CallbackHandler(public_key=settings.langfuse_public_key)
    except Exception as exc:  # noqa: BLE001
        _warn_once(f"could not create callback handler: {exc}")
        return None


# ---------------------------------------------------------------------------
# Explicit spans -- for code LangChain never sees
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def span(name: str, **kwargs: Any) -> Iterator[Any]:
    """Wrap a block in a named span. A no-op when tracing is off.

    Useful for the parts of a run that are not LangGraph nodes -- schema
    retrieval internals, say, or a whole evaluation case.
    """
    client = get_langfuse()
    if client is None:
        yield None
        return
    try:
        with client.start_as_current_observation(
            name=name, as_type="span", **kwargs
        ) as sp:
            yield sp
    except Exception as exc:  # noqa: BLE001
        logger.debug("span %s failed: %s", name, exc)
        yield None


@contextlib.contextmanager
def generation_span(
    name: str,
    model: str,
    input_text: Any = None,
    **kwargs: Any,
) -> Iterator[Any]:
    """Record one model call as a generation. A no-op when tracing is off.

    This closes the gap described at the top of the module: ``llm.py`` talks to
    the provider SDK directly, so the LangChain callback handler never sees the
    actual completions -- only the node that triggered them. Without this, your
    trace shows that ``generate_sql`` ran but not what was sent or returned,
    which is precisely the thing you want to read when a query comes out wrong.

    To wire it up, wrap the provider call in ``llm.complete``::

        with generation_span("generate_sql", model, input_text=user) as gen:
            text, tokens = call(...)
            if gen:
                gen.update(
                    output=text,
                    usage_details={
                        "input": tokens["input"],
                        "output": tokens["output"],
                    },
                )

    Left as an explicit hook rather than applied automatically because
    ``llm.py`` should stay usable with no observability stack present at all.
    """
    client = get_langfuse()
    if client is None:
        yield None
        return
    try:
        with client.start_as_current_observation(
            name=name,
            as_type="generation",
            model=model,
            input=input_text,
            metadata=trace_metadata(),
            **kwargs,
        ) as gen:
            yield gen
    except Exception as exc:  # noqa: BLE001
        logger.debug("generation span %s failed: %s", name, exc)
        yield None


def annotate_current_trace(**fields: Any) -> None:
    """Attach metadata to whatever trace is currently active.

    Called from ``run_agent`` after a question completes, so the trace records
    the outcome -- attempt count, whether self-correction fired -- and you can
    filter the dashboard for "runs that needed a retry" without opening each
    one.
    """
    client = get_langfuse()
    if client is None:
        return
    try:
        client.update_current_span(metadata=trace_metadata(fields))
    except Exception as exc:  # noqa: BLE001
        logger.debug("trace annotation failed: %s", exc)


def trace_url() -> str | None:
    """Link to the current trace, for logs and the demo UI."""
    client = get_langfuse()
    if client is None:
        return None
    try:
        return client.get_trace_url()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# LangSmith alternative
# ---------------------------------------------------------------------------

def enable_langsmith(project: str = "nl2sql-agent") -> bool:
    """Turn on LangSmith instead of Langfuse, if its key is present.

    LangSmith instruments LangChain through environment variables rather than
    an explicit handler, so there is nothing to attach -- setting these before
    the graph is built is the whole integration. Offered because your roadmap
    names either as acceptable; pick one and say which in the README.
    """
    if not os.getenv("LANGCHAIN_API_KEY"):
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    return True


def status() -> dict[str, Any]:
    """Human-readable tracing state, for ``GET /health`` and startup logs."""
    return {
        "langfuse_configured": settings.tracing_enabled,
        "langfuse_client": get_langfuse() is not None,
        "langsmith": bool(os.getenv("LANGCHAIN_TRACING_V2")),
        "run_label": settings.run_label,
        "tags": trace_tags(),
    }


if __name__ == "__main__":  # python -m tracing
    import json

    print(json.dumps(status(), indent=2))
    if settings.tracing_enabled:
        print("auth check:", check_connection())
    else:
        print("\nSet LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env to enable.")