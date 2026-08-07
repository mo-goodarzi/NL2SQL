"""Score a system against the test set and write a reproducible report.

    # Step 2 -- record your "before"
    python eval/run_eval.py --system baseline --label v1_baseline

    # Step 3 -- the agent with self-correction
    python eval/run_eval.py --system agent --label v2_selfcorrect

    # Step 4 -- one flag at a time
    python eval/run_eval.py --system agent --label v3_retrieval --schema-retrieval
    python eval/run_eval.py --system agent --label v4_fewshot --schema-retrieval --few-shot

    # fast iteration while debugging
    python eval/run_eval.py --system baseline --difficulty hard --workers 8

Every row of the accuracy trajectory table in your report comes out of this
file, so two properties matter more than features:

**Reproducibility.** Each report embeds the config flags, prompt version,
model, and git commit that produced it. A number you cannot trace back to a
configuration is a number you cannot defend in an interview.

**Diagnosis.** Every failure records the SQL that was generated and why it was
judged wrong. Reading those failures is how you decide what to build in Step 4
-- it turns "try adding retrieval and see" into "eleven failures reference a
table that was never retrieved, so retrieval should help".
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from compare import compare_case  # noqa: E402
import llm, prompts  # noqa: E402
from config import settings  # noqa: E402
from db import execute_query  # noqa: E402

TESTSET = PROJECT_ROOT / "eval" / "testset.yaml"
REPORTS = PROJECT_ROOT / "eval" / "reports"


# ---------------------------------------------------------------------------
# System under test
# ---------------------------------------------------------------------------

def get_system(name: str) -> Callable[[str], dict[str, Any]]:
    """Return the callable that answers one question.

    Both systems expose the same signature and result shape, which is the
    whole reason the harness does not need to know which one it is scoring.
    """
    if name == "baseline":
        from baseline import answer
        return answer
    if name == "agent":
        try:
            from graph import run_agent
        except ImportError as exc:
            raise SystemExit(
                "The agent is not built yet (src/graph.py missing). "
                "Score --system baseline until Step 3 is done.\n"
                f"  ({exc})"
            ) from exc
        return run_agent
    raise SystemExit(f"Unknown system: {name}")


def git_commit() -> str | None:
    """Best-effort commit hash, so a report points at the exact code."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 -- absence of git is not an error
        return None


# ---------------------------------------------------------------------------
# Scoring one case
# ---------------------------------------------------------------------------

def score_case(case: dict[str, Any], system: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """Run one question and judge it. Never raises.

    Note that the predicted SQL is **re-executed here** rather than reusing the
    rows the system returned. That is deliberate: the agent applies
    ``settings.max_rows`` for the sake of the demo UI, and a truncated result
    compared against a full gold result would mark a correct query wrong.
    Re-running with ``unlimited=True`` also guarantees both sides are measured
    the same way regardless of which system produced them.
    """
    record: dict[str, Any] = {
        "id": case["id"],
        "question": case["question"],
        "difficulty": case["difficulty"],
        "tags": case.get("tags", []),
        "ordered": bool(case.get("ordered", False)),
        "gold_sql": case["gold_sql"].strip(),
        "predicted_sql": None,
        "correct": False,
        "kind": "mismatch",
        "reason": None,
        "attempts": None,
        "failure_class": None,
        "latency_ms": 0.0,
    }

    started = time.perf_counter()

    # --- gold side -------------------------------------------------------
    gold = execute_query(case["gold_sql"], unlimited=True)
    if not gold.ok:
        # A broken gold query is a bug in the test set, not a model failure.
        # Surfaced loudly rather than counted as a miss, because counting it
        # would silently cap the ceiling of every future run.
        record["failure_class"] = "bad_gold_query"
        record["reason"] = f"GOLD QUERY FAILED: {gold.error}"
        record["latency_ms"] = (time.perf_counter() - started) * 1000
        return record

    # --- system side -----------------------------------------------------
    try:
        out = system(case["question"])
    except Exception as exc:  # noqa: BLE001 -- one bad case must not kill the run
        record["failure_class"] = "system_exception"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        record["latency_ms"] = (time.perf_counter() - started) * 1000
        return record

    record["predicted_sql"] = out.get("sql")
    record["attempts"] = out.get("attempts")

    if out.get("unanswerable"):
        record["failure_class"] = "unanswerable"
        record["reason"] = "system declined: unanswerable from this schema"
    elif not out.get("sql"):
        record["failure_class"] = "no_sql"
        record["reason"] = out.get("error") or "no SQL produced"
    else:
        pred = execute_query(out["sql"], unlimited=True)
        if not pred.ok:
            record["failure_class"] = "execution_error"
            record["reason"] = pred.error
        else:
            cmp = compare_case(
                case, gold.columns, gold.rows, pred.columns, pred.rows
            )
            record.update(
                correct=cmp.match,
                kind=cmp.kind,
                reason=cmp.reason,
                both_empty=cmp.both_empty,
                near_miss=cmp.near_miss,
                order_only=cmp.order_only,
            )
            if not cmp.match:
                record["failure_class"] = (
                    "wrong_order" if cmp.order_only
                    else "rounding" if cmp.near_miss
                    else "wrong_result"
                )

    record["latency_ms"] = (time.perf_counter() - started) * 1000
    return record


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    correct = sum(1 for r in records if r["correct"])

    by_diff: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in records:
        d = by_diff[r["difficulty"]]
        d["n"] += 1
        d["correct"] += int(r["correct"])

    by_tag: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in records:
        for t in r["tags"]:
            by_tag[t]["n"] += 1
            by_tag[t]["correct"] += int(r["correct"])

    for bucket in (by_diff, by_tag):
        for v in bucket.values():
            v["accuracy"] = round(v["correct"] / v["n"], 4) if v["n"] else 0.0

    latencies = sorted(r["latency_ms"] for r in records)
    return {
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "by_difficulty": dict(by_diff),
        "by_tag": dict(by_tag),
        "failure_classes": dict(Counter(
            r["failure_class"] for r in records if r["failure_class"]
        )),
        "matched_by_permutation": sum(1 for r in records if r.get("kind") == "permuted"),
        "both_empty": sum(1 for r in records if r.get("both_empty")),
        "mean_latency_ms": round(sum(latencies) / n, 1) if n else 0.0,
        "p95_latency_ms": round(latencies[int(n * 0.95)], 1) if n else 0.0,
    }


def print_report(records: list[dict[str, Any]], summary: dict[str, Any], label: str) -> None:
    acc = summary["accuracy"]
    print(f"\n{'=' * 72}")
    print(f"  {label}:  {acc:.1%}  ({summary['correct']}/{summary['n']})")
    print(f"{'=' * 72}")

    print("\n  by difficulty")
    for d in ("easy", "medium", "hard"):
        if d in summary["by_difficulty"]:
            v = summary["by_difficulty"][d]
            bar = "#" * int(v["accuracy"] * 24)
            print(f"    {d:<8} {v['accuracy']:>6.1%}  {v['correct']:>2}/{v['n']:<3} {bar}")

    weak = sorted(
        (t for t, v in summary["by_tag"].items() if v["n"] >= 2),
        key=lambda t: summary["by_tag"][t]["accuracy"],
    )[:6]
    if weak:
        print("\n  weakest tags (these tell you what to fix next)")
        for t in weak:
            v = summary["by_tag"][t]
            print(f"    {t:<20} {v['accuracy']:>6.1%}  {v['correct']}/{v['n']}")

    if summary["failure_classes"]:
        print("\n  failure classes")
        for k, v in sorted(summary["failure_classes"].items(), key=lambda x: -x[1]):
            print(f"    {k:<20} {v}")

    failures = [r for r in records if not r["correct"]]
    if failures:
        print(f"\n  failures ({len(failures)})")
        for r in failures:
            print(f"\n    {r['id']} [{r['difficulty']}] {r['question']}")
            print(f"      why: {r['failure_class']} -- {str(r['reason'])[:140]}")
            if r["predicted_sql"]:
                sql = " ".join(r["predicted_sql"].split())
                print(f"      got: {sql[:160]}")

    if summary["both_empty"]:
        print(f"\n  NOTE {summary['both_empty']} case(s) matched with both sides empty.")
        print("       Review these -- an empty gold result cannot distinguish a")
        print("       correct query from a broken one.")
    if summary["matched_by_permutation"]:
        print(f"\n  NOTE {summary['matched_by_permutation']} case(s) matched only after "
              "reordering columns.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--system", choices=["baseline", "agent"], default="baseline")
    p.add_argument("--label", default=None, help="Run name for the report file.")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel questions. Lower this if you hit rate limits.")
    p.add_argument("--model", default=None, help="Override settings.model.")

    # Step 4 toggles, so a sweep needs no code or .env edits.
    p.add_argument("--schema-retrieval", action="store_true")
    p.add_argument("--few-shot", action="store_true")
    p.add_argument("--no-self-correction", action="store_true")
    p.add_argument("--max-retries", type=int, default=None)

    # Subsetting, for fast iteration.
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)

    p.add_argument("--strict-columns", action="store_true",
                   help="Do not accept a match that needs column reordering.")
    p.add_argument("--no-report", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Apply overrides before importing the system, so it picks them up.
    if args.model:
        settings.model = args.model
    if args.schema_retrieval:
        settings.use_schema_retrieval = True
    if args.few_shot:
        settings.use_few_shot = True
    if args.no_self_correction:
        settings.use_self_correction = False
    if args.max_retries is not None:
        settings.max_retries = args.max_retries

    label = args.label or f"{args.system}_{datetime.now():%Y%m%d_%H%M%S}"
    settings.run_label = label
    settings.require_database()

    cases = yaml.safe_load(TESTSET.read_text())
    if args.difficulty:
        cases = [c for c in cases if c["difficulty"] == args.difficulty]
    if args.tag:
        cases = [c for c in cases if args.tag in c.get("tags", [])]
    if args.ids:
        cases = [c for c in cases if c["id"] in set(args.ids)]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No cases selected.")

    system = get_system(args.system)
    llm.reset_usage()

    print(f"Running {len(cases)} case(s) | system={args.system} | model={settings.model}")
    print(f"flags: {settings.active_flags}")

    started = time.perf_counter()
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(lambda c: score_case(c, system), cases))
    else:
        records = [score_case(c, system) for c in cases]
    duration = time.perf_counter() - started

    bad_gold = [r["id"] for r in records if r["failure_class"] == "bad_gold_query"]
    if bad_gold:
        print(f"\n  !! {len(bad_gold)} GOLD QUERIES FAILED: {', '.join(bad_gold)}")
        print("     Fix these before trusting the score -- run "
              "eval/validate_testset.py")

    summary = summarise(records)
    print_report(records, summary, label)

    usage = llm.usage_summary()
    print(f"\n  cost: ${usage['cost_usd']:.4f} over {usage['calls']} model call(s)"
          + ("  (incomplete -- unknown model pricing)" if usage["cost_incomplete"] else ""))
    print(f"  time: {duration:.1f}s wall, {summary['mean_latency_ms']:.0f}ms mean/question")

    if not args.no_report:
        REPORTS.mkdir(parents=True, exist_ok=True)
        path = REPORTS / f"{datetime.now():%Y%m%d_%H%M%S}_{label}.json"
        path.write_text(json.dumps({
            "label": label,
            "system": args.system,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(duration, 1),
            "git_commit": git_commit(),
            "config": settings.active_flags,
            "prompt": prompts.prompt_metadata(),
            "usage": usage,
            "summary": summary,
            "cases": records,
        }, indent=2, default=str))
        print(f"\n  report: {path.relative_to(PROJECT_ROOT)}")

    print(f"\n  trajectory row:  | {label} | {summary['accuracy']:.0%} | "
          f"${usage['cost_usd']:.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())