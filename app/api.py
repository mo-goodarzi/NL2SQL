"""HTTP surface over the agent.

A thin layer on purpose: it validates input, calls ``graph.run_agent``, and
shapes the response. All the interesting behaviour lives in the library, which
is what lets ``eval/run_eval.py`` score exactly the code this serves.

Two details that are easy to get wrong and expensive to debug:

**The endpoints are ``def``, not ``async def``.** ``run_agent`` is fully
blocking -- provider HTTP calls and SQLite queries. Declaring it ``async``
would run it on the event loop and freeze every other request for the duration
of a model call, which on a single-worker free-tier host means the demo serves
one visitor at a time. A plain ``def`` endpoint runs in FastAPI's thread pool
instead, and ``db.py`` already uses thread-local connections, so concurrency
is safe.

**The SQL is returned even when the query failed.** Visible SQL is what makes
the agent legible rather than a black box, and a reviewer clicking your demo
looks for it first. Hiding it on failure hides the most interesting case.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Make ``src`` importable when this app is run directly or via uvicorn, same
# as scripts/load_data.py and eval/run_eval.py do for their own entry points.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import schema, tracing  # noqa: E402
from config import settings  # noqa: E402
from db import health_check  # noqa: E402
from graph import run_agent  # noqa: E402

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500

#: Requests per IP per window. This is a public demo with your API key behind
#: it, so an unthrottled endpoint is an invitation to spend your credits. A
#: process-local counter is not a real rate limiter -- it resets on restart and
#: does not span replicas -- but it stops casual abuse, which is the actual
#: threat to a portfolio demo.
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW_S = 60

_request_log: dict[str, deque[float]] = defaultdict(deque)

EXAMPLE_QUESTIONS = [
    "How many orders are in the database?",
    "What are the top 5 states by number of customers?",
    "Which products are in the health_beauty category?",
    "What is the total revenue for each product category?",
    "What is the average review score by month in 2017?",
]


# ---------------------------------------------------------------------------
# Schemas -- these generate the OpenAPI docs page, which is free credibility
# on a demo, so the field descriptions are written to be read.
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=MAX_QUESTION_CHARS,
        description="A natural-language question about the e-commerce data.",
        examples=["What are the top 5 states by number of customers?"],
    )


class QueryResponse(BaseModel):
    ok: bool = Field(description="Whether a result was successfully produced.")
    question: str
    sql: str | None = Field(
        None,
        description="The generated SQL. Returned even on failure, so the "
                    "attempt is always inspectable.",
    )
    columns: list[str] | None = None
    rows: list[list[Any]] = Field(
        default_factory=list,
        description=f"Result rows, capped at {settings.max_rows}.",
    )
    row_count: int = 0
    explanation: str | None = Field(
        None, description="One-sentence natural-language answer."
    )
    attempts: int = Field(
        0, description="Query attempts made. Above 1 means the agent "
                       "self-corrected after a failure."
    )
    self_corrected: bool = False
    error: str | None = None
    unanswerable: bool = Field(
        False, description="The question cannot be answered from this schema."
    )
    retrieved_tables: list[str] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-attempt trace: the SQL tried and why it failed.",
    )
    elapsed_ms: float = 0.0
    trace_url: str | None = None


class HealthResponse(BaseModel):
    ok: bool
    database: dict[str, Any]
    model: str
    api_key_configured: bool
    tracing: dict[str, Any]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fail loudly at startup rather than on the first visitor's request."""
    try:
        settings.require_database()
        logger.info("database ok: %s", settings.db_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
    try:
        settings.require_api_key()
    except RuntimeError as exc:
        logger.error("%s", exc)
    logger.info("tracing: %s", tracing.status())
    yield
    tracing.flush()  # a container stopping must not discard buffered traces


app = FastAPI(
    title="Text-to-SQL Analytics Agent",
    description=(
        "Ask questions in plain English about a Brazilian e-commerce dataset. "
        "The agent inspects the schema, writes SQL, runs it read-only, and "
        "corrects itself if the query fails."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# The Streamlit UI is served from a different origin than the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _rate_limited(request: Request) -> bool:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = _request_log[ip]
    while window and now - window[0] > RATE_LIMIT_WINDOW_S:
        window.popleft()
    if len(window) >= RATE_LIMIT_REQUESTS:
        return True
    window.append(now)
    return False


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Return a clean payload instead of a stack trace.

    A traceback in an HTTP response leaks file paths and library versions, and
    tells a visitor nothing useful. Logged in full, summarised to the client.
    """
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": "Internal error. Please try again."},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query(request_body: QueryRequest, request: Request) -> QueryResponse:
    """Answer one natural-language question.

    Declared ``def`` rather than ``async def`` deliberately -- see the module
    docstring. The blocking work runs in FastAPI's thread pool.
    """
    if _rate_limited(request):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: {RATE_LIMIT_REQUESTS} requests per "
                   f"{RATE_LIMIT_WINDOW_S}s. Please wait a moment.",
        )

    question = request_body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    started = time.perf_counter()
    result = run_agent(question)
    elapsed = (time.perf_counter() - started) * 1000

    return QueryResponse(
        ok=result["ok"],
        question=question,
        sql=result["sql"],
        columns=result["columns"],
        # Tuples are not JSON; convert once here rather than trusting the
        # serialiser to do something sensible with them.
        rows=[list(r) for r in result["rows"]],
        row_count=result["row_count"],
        explanation=result["explanation"],
        attempts=result["attempts"],
        self_corrected=result["attempts"] > 1,
        error=result["error"],
        unanswerable=result["unanswerable"],
        retrieved_tables=result.get("retrieved_tables", []),
        history=result["history"],
        elapsed_ms=round(elapsed, 1),
        trace_url=tracing.trace_url(),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Readiness probe for the deployment platform and the UI's status badge."""
    db = health_check()
    try:
        settings.require_api_key()
        key_ok = True
    except RuntimeError:
        key_ok = False

    return HealthResponse(
        ok=bool(db.get("ok")) and key_ok,
        database=db,
        model=settings.model,
        api_key_configured=key_ok,
        tracing=tracing.status(),
    )


@app.get("/schema")
def get_schema() -> dict[str, Any]:
    """Tables and columns, so the UI can show users what they can ask about.

    A visitor who cannot see the data model asks questions the schema cannot
    answer, then concludes the agent is broken.
    """
    tables = schema.get_tables()
    return {
        "tables": [
            {
                "name": name,
                "columns": [
                    {"name": c.name, "type": c.type, "primary_key": c.primary_key}
                    for c in table.columns
                ],
                "foreign_keys": [
                    {"column": fk.column, "references":
                     f"{fk.ref_table}.{fk.ref_column}"}
                    for fk in table.foreign_keys
                ],
            }
            for name, table in tables.items()
        ]
    }


@app.get("/examples")
def examples() -> dict[str, list[str]]:
    """Starter questions. An empty text box is the worst possible demo."""
    return {"examples": EXAMPLE_QUESTIONS}


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "Text-to-SQL Analytics Agent",
        "docs": "/docs",
        "endpoints": ["/query", "/health", "/schema", "/examples"],
    }


if __name__ == "__main__":  # python -m app.api
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)