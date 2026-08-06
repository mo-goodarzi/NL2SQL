"""The deliberately dumb single-shot system: your experimental control.

Dump the whole schema into one prompt, ask for SQL, run it once, return the
result. No retries, no schema retrieval, no few-shot, no validation. That is
the entire design, and every omission is intentional.

**Do not improve this file.** Every improvement you sneak in here shrinks the
gap your evaluation report is trying to demonstrate. If you find a better
prompt or a smarter schema rendering, it belongs in the agent, where it can be
measured as a change. The baseline's whole job is to stay bad and stay
comparable.

Two things keep it comparable:

* **It builds its own naive schema dump** rather than importing ``schema.py``.
  When you add foreign-key-aware retrieval in Step 4, that improvement must not
  quietly leak backwards into the control and erase the very lift you are
  trying to show.
* **It ignores the Step 4 config toggles.** ``use_few_shot`` and
  ``use_schema_retrieval`` do nothing here. Sweeping those flags changes the
  agent and leaves the baseline fixed, which is what makes the two columns of
  your table comparable.

One thing that does *not* stay fixed: the prompt text in ``prompts.py``. This
module shares it, so a single source of truth for prompt wording is preserved
-- but it means **editing prompts.py invalidates your recorded baseline**.
Re-run the baseline after any prompt change. The returned ``prompt_version``
field exists so a stale number is at least detectable.
"""

from __future__ import annotations

import argparse
import logging
from functools import lru_cache
from typing import Any

import prompts
from config import settings
from db import execute_query, fetch_all
from llm import LLMError, complete, extract_sql

logger = logging.getLogger(__name__)

SYSTEM_NAME = "baseline"


@lru_cache(maxsize=1)
def naive_schema_text() -> str:
    """Every CREATE TABLE statement, verbatim, in one blob.

    This is the naive approach on purpose: no relevance filtering, no sample
    values, no compaction -- just what ``sqlite_master`` already stores. It is
    exactly the "dump the whole schema and hope" strategy that the agent's
    retrieval mode is meant to beat.

    Deliberately does not import ``schema.py``. Keeping the control's schema
    rendering frozen is what stops a later improvement from raising the
    baseline and hiding your own gains.
    """
    rows = fetch_all(
        "SELECT sql FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
        "ORDER BY name"
    )
    return "\n\n".join(r[0] for r in rows)


def answer(question: str, model: str | None = None) -> dict[str, Any]:
    """Answer one question with a single model call and a single query.

    Returns the same dict shape as the agent's ``run_agent`` so
    ``eval/run_eval.py`` can score either system without branching. Fields the
    baseline cannot produce are present but empty: ``attempts`` is always 1,
    ``history`` is always empty, and ``explanation`` is always None because
    the baseline does not do a second call to explain itself.

    Never raises. A model or database failure comes back as ``ok=False`` with
    the reason in ``error`` -- a fifty-question evaluation run must not die on
    question seventeen.
    """
    result: dict[str, Any] = {
        "system": SYSTEM_NAME,
        "question": question,
        "sql": None,
        "ok": False,
        "columns": None,
        "rows": [],
        "row_count": 0,
        "error": None,
        "explanation": None,
        "unanswerable": False,
        "attempts": 1,
        "history": [],
        "prompt_version": prompts.PROMPT_VERSION,
        "model": model or settings.model,
    }

    # --- 1. Ask for SQL -----------------------------------------------------
    try:
        schema_text = naive_schema_text()
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"Schema introspection failed: {exc}"
        return result

    try:
        raw = complete(
            system=prompts.SQL_SYSTEM,
            # examples=[] hard-forces few-shot off regardless of config. The
            # control must not move when a Step 4 flag is flipped.
            user=prompts.build_sql_prompt(question, schema_text, examples=[]),
            model=model,
        )
    except LLMError as exc:
        result["error"] = f"Model call failed: {exc}"
        return result

    # --- 2. Handle the two non-SQL outcomes --------------------------------
    # UNANSWERABLE is checked against the raw response, before extraction.
    # extract_sql looks for SELECT/WITH and would strip this to an empty
    # string, mislabelling a deliberate refusal as "no SQL returned".
    if raw.strip().upper().startswith("UNANSWERABLE"):
        # The escape hatch offered by SQL_SYSTEM. A clean terminal state, not
        # an error to retry -- and a signal your test set may contain a
        # question the schema genuinely cannot answer.
        result["error"] = "Model judged the question unanswerable from this schema."
        result["unanswerable"] = True
        return result

    sql = extract_sql(raw)

    if not sql:
        # extract_sql returns empty when the response contains no SQL at all.
        # Distinct from bad SQL, and worth counting separately in your report.
        result["error"] = "Model returned no SQL."
        result["sql"] = raw.strip()[:500] or None
        return result

    result["sql"] = sql

    # --- 3. Run it once. No validation, no retry. --------------------------
    # Note there is no guardrails check here: guardrails.py is a Step 3 module
    # and belongs to the agent. Safety still holds because db.py opens the
    # connection read-only, so the worst a bad statement can do is fail.
    query = execute_query(sql, unlimited=True)

    result["ok"] = query.ok
    result["columns"] = query.columns
    result["rows"] = query.rows
    result["row_count"] = query.row_count
    result["error"] = query.error
    result["elapsed_ms"] = query.elapsed_ms
    return result


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Ask the baseline one question.")
    parser.add_argument("question", help="Natural-language question")
    parser.add_argument("--model", default=None, help="Override settings.model")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    out = answer(args.question, model=args.model)

    print(f"\nSQL:\n{out['sql']}\n")
    if out["ok"]:
        print(f"{out['row_count']} row(s)")
        if out["columns"]:
            print(" | ".join(out["columns"]))
        for row in out["rows"][:10]:
            print(" | ".join("NULL" if v is None else str(v) for v in row))
        if out["row_count"] > 10:
            print(f"... and {out['row_count'] - 10} more")
    else:
        print(f"FAILED: {out['error']}")

    from llm import usage_summary
    print(f"\nusage: {usage_summary()}")


if __name__ == "__main__":  # python baseline.py "how many orders?"
    _cli()