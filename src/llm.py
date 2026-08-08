"""Thin wrapper over the model provider.

Everything that knows about SDKs, retries, or token costs lives here. Two
reasons that matters:

* **Swapping or comparing models becomes a config change.** ``prompts.py`` and
  ``nodes.py`` never import a vendor SDK, so running the same evaluation
  against a cheaper model is one environment variable, not a refactor.
* **Cost lands in one place.** Being able to say "83% accuracy at $0.004 per
  question" is unusually strong signal for a junior portfolio -- most projects
  report accuracy and have no idea what they spent.

Note the deliberate split between two kinds of retry. This module retries
*transport* failures (rate limits, 5xx, dropped connections) because they say
nothing about the answer. It does **not** retry bad SQL -- that is the agent's
self-correction loop in ``nodes.py``, which is a semantic retry with new
information in the prompt. Conflating the two would make your evaluation
numbers meaningless, because a "retry" would sometimes mean "the network
hiccuped" and sometimes "the model learned from an error".
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing
#
# USD per million tokens (input, output), verified against
# https://platform.claude.com/docs/en/about-claude/pricing on 2026-08-05.
#
# Prices change. Treat this table as an estimate for relative comparison
# between runs, not as a billing source of truth -- the Console is that. An
# unknown model yields a null cost rather than a fabricated one, because a
# silently wrong number in an evaluation report is worse than a missing one.
# ---------------------------------------------------------------------------

PRICING_VERIFIED_ON = "2026-08-05"

PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    # claude-sonnet-5 is handled by _sonnet_5_rates() -- see below.
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    # OpenAI, verified against https://platform.openai.com/docs/pricing on
    # PRICING_VERIFIED_ON. Cached-input tokens are 10% of the input rate for
    # all of these too, so CACHE_READ_MULTIPLIER applies unchanged.
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "o3": (2.0, 8.0),
    "o4-mini": (1.10, 4.40),
}

#: Sonnet 5 runs on introductory pricing that expires. Encoding the date rather
#: than hardcoding one rate keeps cost figures honest across a project that
#: spans the changeover.
_SONNET_5_INTRO_UNTIL = date(2026, 8, 31)
_SONNET_5_INTRO = (2.0, 10.0)
_SONNET_5_STANDARD = (3.0, 15.0)

#: Multipliers relative to the base input rate, from the same pricing page.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_5M_MULTIPLIER = 1.25


def _sonnet_5_rates(on: date | None = None) -> tuple[float, float]:
    on = on or date.today()
    return _SONNET_5_INTRO if on <= _SONNET_5_INTRO_UNTIL else _SONNET_5_STANDARD


def get_rates(model: str, on: date | None = None) -> tuple[float, float] | None:
    """Input/output USD per million tokens, or None if the model is unknown."""
    base = model.split("@")[0]  # tolerate Vertex-style suffixes
    if base.startswith("claude-sonnet-5"):
        return _sonnet_5_rates(on)
    if base in PRICING:
        return PRICING[base]
    # Dated aliases like claude-opus-5-20260724 -> longest matching prefix.
    matches = [k for k in PRICING if base.startswith(k)]
    return PRICING[max(matches, key=len)] if matches else None


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    """Accumulated token usage and estimated spend."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    #: True if any call used a model missing from PRICING, meaning cost_usd is
    #: an undercount. Surfaced in reports so a low number is never mistaken
    #: for a cheap run.
    cost_incomplete: bool = False
    latency_ms: list[float] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def mean_latency_ms(self) -> float:
        return sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "cost_incomplete": self.cost_incomplete,
            "mean_latency_ms": round(self.mean_latency_ms, 1),
        }

    def __str__(self) -> str:
        flag = " (incomplete)" if self.cost_incomplete else ""
        return (
            f"{self.calls} calls | {self.input_tokens:,} in / "
            f"{self.output_tokens:,} out | ${self.cost_usd:.4f}{flag} | "
            f"{self.mean_latency_ms:.0f}ms avg"
        )


class UsageTracker:
    """Thread-safe accumulator for one run.

    The evaluation harness resets this before each run and snapshots it after,
    so every row of the accuracy trajectory table can carry its own cost and
    latency alongside the score.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usage = Usage()

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> float:
        """Add one call's usage; returns that call's estimated cost."""
        rates = get_rates(model)
        if rates is None:
            cost = 0.0
        else:
            in_rate, out_rate = rates
            cost = (
                input_tokens * in_rate
                + output_tokens * out_rate
                + cache_read_tokens * in_rate * CACHE_READ_MULTIPLIER
                + cache_write_tokens * in_rate * CACHE_WRITE_5M_MULTIPLIER
            ) / 1_000_000

        with self._lock:
            u = self._usage
            u.calls += 1
            u.input_tokens += input_tokens
            u.output_tokens += output_tokens
            u.cache_read_tokens += cache_read_tokens
            u.cache_write_tokens += cache_write_tokens
            u.cost_usd += cost
            u.latency_ms.append(latency_ms)
            if rates is None:
                u.cost_incomplete = True
        return cost

    def snapshot(self) -> Usage:
        with self._lock:
            u = self._usage
            return Usage(
                calls=u.calls,
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cache_read_tokens=u.cache_read_tokens,
                cache_write_tokens=u.cache_write_tokens,
                cost_usd=u.cost_usd,
                cost_incomplete=u.cost_incomplete,
                latency_ms=list(u.latency_ms),
            )

    def reset(self) -> None:
        with self._lock:
            self._usage = Usage()


usage = UsageTracker()


# ---------------------------------------------------------------------------
# Errors and retry policy
# ---------------------------------------------------------------------------

class LLMError(RuntimeError):
    """A model call that could not be completed, after retries."""


#: Transport-level failures worth retrying. 429 is rate limiting; 5xx are the
#: provider's problem; 408/409 are transient. Anything else (400 bad request,
#: 401 auth, 403, 404, 422) is a bug in our request and retrying just burns
#: quota while producing the same error.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
_RETRYABLE_NAMES = {
    "APIConnectionError", "APITimeoutError", "ConnectionError",
    "InternalServerError", "RateLimitError", "OverloadedError",
    "ServiceUnavailableError", "Timeout", "TimeoutError",
}


def _is_retryable(exc: Exception) -> bool:
    """Classify by status code and class name, not by SDK exception imports.

    Keeps the policy identical across providers and avoids a hard dependency
    on any one SDK's exception hierarchy.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS
    return type(exc).__name__ in _RETRYABLE_NAMES


def _retry_after_seconds(exc: Exception) -> float | None:
    """Honour a server-supplied Retry-After header when there is one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _backoff(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Exponential backoff with full jitter.

    Jitter matters when the evaluation harness runs questions concurrently: a
    fixed schedule makes every worker retry in lockstep and re-trigger the
    same rate limit.
    """
    return random.uniform(0, min(cap, base * (2 ** attempt)))


# ---------------------------------------------------------------------------
# Provider clients (imported lazily so the SDK is only needed when used)
# ---------------------------------------------------------------------------

_client_lock = threading.Lock()
_client: Any = None


def _get_client() -> Any:
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        key = settings.require_api_key()
        if settings.llm_provider == "anthropic":
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise LLMError("pip install anthropic") from exc
            _client = anthropic.Anthropic(api_key=key)
        else:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover
                raise LLMError("pip install openai") from exc
            _client = openai.OpenAI(api_key=key)
    return _client


def _call_anthropic(
    client: Any, model: str, system: str, user: str,
    temperature: float, max_tokens: int, stop: list[str] | None,
) -> tuple[str, dict[str, int]]:
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
        **({"stop_sequences": stop} if stop else {}),
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    u = resp.usage
    return text, {
        "input": getattr(u, "input_tokens", 0) or 0,
        "output": getattr(u, "output_tokens", 0) or 0,
        "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


#: Reasoning-family models (gpt-5*, o1/o3/o4*) reject any temperature other
#: than their fixed default (1) -- confirmed empirically, not documented
#: consistently. settings.temperature (0.0, for reproducible SQL generation)
#: is silently unenforceable on these; callers should treat eval runs against
#: them as non-deterministic.
_FIXED_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _supports_temperature(model: str) -> bool:
    return not model.startswith(_FIXED_TEMPERATURE_PREFIXES)


def _call_openai(
    client: Any, model: str, system: str, user: str,
    temperature: float, max_tokens: int, stop: list[str] | None,
) -> tuple[str, dict[str, int]]:
    resp = client.chat.completions.create(
        model=model,
        # max_tokens is deprecated across the Chat Completions API and
        # rejected outright by reasoning-family models; max_completion_tokens
        # works on every model tested, old and new.
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **({"temperature": temperature} if _supports_temperature(model) else {}),
        **({"stop": stop} if stop else {}),
    )
    text = resp.choices[0].message.content or ""
    u = resp.usage
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return text, {
        "input": (getattr(u, "prompt_tokens", 0) or 0) - cached,
        "output": getattr(u, "completion_tokens", 0) or 0,
        "cache_read": cached,
        "cache_write": 0,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def complete(
    system: str,
    user: str,
    temperature: float | None = None,
    max_tokens: int = 1024,
    stop: list[str] | None = None,
    model: str | None = None,
    max_attempts: int = 4,
) -> str:
    """Send one prompt and return the model's text.

    Args:
        system: System prompt.
        user: User message.
        temperature: Defaults to ``settings.temperature`` (0.0). Leave it there
            for SQL generation -- sampling noise will move your evaluation
            score by a few points between identical runs and you will not be
            able to tell noise from a real improvement.
        max_tokens: Output ceiling. SQL is short; 1024 is generous.
        stop: Optional stop sequences.
        model: Override ``settings.model`` for A/B comparisons.
        max_attempts: Transport retries, not semantic ones.

    Raises:
        LLMError: after exhausting retries, or immediately on a
            non-retryable error such as a malformed request or bad key.
    """
    model = model or settings.model
    temperature = settings.temperature if temperature is None else temperature
    client = _get_client()
    call: Callable[..., tuple[str, dict[str, int]]] = (
        _call_anthropic if settings.llm_provider == "anthropic" else _call_openai
    )

    last_exc: Exception | None = None
    attempts_made = 0
    for attempt in range(max_attempts):
        started = time.perf_counter()
        attempts_made = attempt + 1
        try:
            text, tokens = call(
                client, model, system, user, temperature, max_tokens, stop
            )
        except Exception as exc:  # noqa: BLE001 -- classified below
            last_exc = exc
            if not _is_retryable(exc) or attempt == max_attempts - 1:
                break
            delay = _retry_after_seconds(exc) or _backoff(attempt)
            logger.warning(
                "LLM call failed (%s), retrying in %.1fs [%d/%d]",
                type(exc).__name__, delay, attempt + 1, max_attempts - 1,
            )
            time.sleep(delay)
            continue

        elapsed_ms = (time.perf_counter() - started) * 1000
        cost = usage.record(
            model=model,
            input_tokens=tokens["input"],
            output_tokens=tokens["output"],
            cache_read_tokens=tokens["cache_read"],
            cache_write_tokens=tokens["cache_write"],
            latency_ms=elapsed_ms,
        )
        logger.debug(
            "%s: %d in / %d out, $%.5f, %.0fms",
            model, tokens["input"], tokens["output"], cost, elapsed_ms,
        )
        return text

    raise LLMError(
        f"{type(last_exc).__name__} after {attempts_made} attempt(s): {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# SQL extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_START_RE = re.compile(r"\b(WITH|SELECT)\b", re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull a bare SQL statement out of a model response.

    Models are told to return only SQL and mostly comply, but not always: they
    wrap it in fences, preface it with "Here is the query:", or append an
    explanation. Rather than fight that with prompt engineering alone, normalise
    it here -- it is cheaper and more reliable than another retry.

    Trailing prose is cut at the first semicolon, which also means only the
    first statement survives. That pairs with ``guardrails.py`` rejecting
    multi-statement input: a model appending ``; DROP TABLE`` loses the second
    statement here and would be refused there anyway.

    Returns an empty string when the response contains no SQL at all (a refusal,
    or a clarifying question). That is deliberately distinguishable from
    returning bad SQL: they are different failure modes and your evaluation
    report should count them separately.
    """
    if not text:
        return ""

    fence = _FENCE_RE.search(text)
    candidate = fence.group(1) if fence else text

    match = _START_RE.search(candidate)
    if not match:
        return ""
    candidate = candidate[match.start():]

    # Everything after the first statement terminator is prose, not SQL.
    semicolon = candidate.find(";")
    if semicolon != -1:
        candidate = candidate[:semicolon]

    return candidate.strip()


def reset_usage() -> None:
    """Convenience for the evaluation harness between runs."""
    usage.reset()


def usage_summary() -> dict[str, Any]:
    """Current run's usage, for stamping onto reports and traces."""
    snap = usage.snapshot()
    return {
        **snap.to_dict(),
        "model": settings.model,
        "pricing_verified_on": PRICING_VERIFIED_ON,
    }