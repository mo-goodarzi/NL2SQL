"""Check the test set before you trust a score from it.

    python eval/validate_testset.py
    python eval/validate_testset.py --show q030 q038

A wrong gold query silently caps your accuracy: the agent produces a correct
answer, it disagrees with a broken gold result, and you spend an afternoon
debugging an agent that was never at fault. Run this after every edit to
testset.yaml, and read the flagged output before recording a baseline.

What it checks automatically:

* structure -- unique ids, required fields, valid difficulty values
* execution -- every gold query runs against the real database
* smells   -- queries returning zero rows, or a single NULL, which usually
              mean a filter that matches nothing rather than a real answer
* ordering -- any query with LIMIT should be marked ``ordered: true``, and
              should have a deterministic tiebreak in its ORDER BY

What it cannot check is whether a query answers the question it is paired
with. Only you can do that, which is what ``--show`` is for.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import settings  # noqa: E402

TESTSET = PROJECT_ROOT / "eval" / "testset.yaml"
REQUIRED = {"id", "question", "difficulty", "tags", "gold_sql"}
DIFFICULTIES = {"easy", "medium", "hard"}


def load_cases() -> list[dict]:
    cases = yaml.safe_load(TESTSET.read_text())
    if not isinstance(cases, list):
        raise SystemExit(f"{TESTSET} must contain a top-level list.")
    return cases


def check_structure(cases: list[dict]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for i, c in enumerate(cases):
        cid = c.get("id", f"<index {i}>")
        missing = REQUIRED - set(c)
        if missing:
            problems.append(f"{cid}: missing field(s) {sorted(missing)}")
        if cid in seen:
            problems.append(f"{cid}: duplicate id")
        seen.add(cid)
        if c.get("difficulty") not in DIFFICULTIES:
            problems.append(f"{cid}: difficulty must be one of {sorted(DIFFICULTIES)}")
        if not c.get("tags"):
            problems.append(f"{cid}: at least one tag required")
    return problems


def check_ordering(cases: list[dict]) -> list[str]:
    """LIMIT without ordered:true is the most common test-set authoring bug."""
    warnings: list[str] = []
    for c in cases:
        sql = c.get("gold_sql", "")
        has_limit = re.search(r"\bLIMIT\b", sql, re.IGNORECASE)
        if has_limit and not c.get("ordered"):
            warnings.append(
                f"{c['id']}: has LIMIT but ordered is not true -- row order "
                f"will not be compared, so a wrongly-sorted answer may pass"
            )
        if has_limit:
            order_by = re.search(r"ORDER BY(.+?)(?:LIMIT|$)", sql, re.IGNORECASE | re.DOTALL)
            if order_by and "," not in order_by.group(1):
                warnings.append(
                    f"{c['id']}: ORDER BY has no tiebreak column -- ties at the "
                    f"LIMIT boundary make the expected rows nondeterministic"
                )
    return warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", nargs="*", default=[], metavar="ID",
                    help="Print full results for these case ids, for eyeballing.")
    ap.add_argument("--rows", type=int, default=5, help="Rows to print with --show.")
    args = ap.parse_args()

    cases = load_cases()
    print(f"{TESTSET.relative_to(PROJECT_ROOT)}: {len(cases)} cases\n")

    problems = check_structure(cases)
    if problems:
        print("STRUCTURE PROBLEMS")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Structure OK")

    spread = Counter(c["difficulty"] for c in cases)
    print(f"  difficulty: " + ", ".join(f"{k} {spread[k]}" for k in ("easy", "medium", "hard")))
    print(f"  ordered:    {sum(1 for c in cases if c.get('ordered'))}")
    tags = Counter(t for c in cases for t in c["tags"])
    print(f"  tags:       {len(tags)} distinct, top: "
          + ", ".join(f"{t}({n})" for t, n in tags.most_common(6)))

    settings.require_database()
    conn = sqlite3.connect(settings.read_only_uri, uri=True)

    failures, empty, null_only = [], [], []
    for c in cases:
        try:
            cur = conn.execute(c["gold_sql"])
            rows = cur.fetchall()
        except sqlite3.Error as exc:
            failures.append((c["id"], str(exc)))
            continue
        if not rows:
            empty.append(c["id"])
        elif len(rows) == 1 and all(v is None for v in rows[0]):
            null_only.append(c["id"])

    print(f"\nExecution: {len(cases) - len(failures)}/{len(cases)} ran successfully")
    for cid, err in failures:
        print(f"  FAIL {cid}: {err}")

    if empty:
        print(f"\n  Zero rows ({len(empty)}): {', '.join(empty)}")
        print("    Usually a filter matching nothing. Verify each is genuinely correct --")
        print("    an empty gold result makes a broken agent query look right.")
    if null_only:
        print(f"\n  Single NULL result ({len(null_only)}): {', '.join(null_only)}")
        print("    Often an aggregate over an empty set. Check the filter.")

    for w in check_ordering(cases):
        print(f"\n  WARN {w}")

    by_id = {c["id"]: c for c in cases}
    for cid in args.show:
        c = by_id.get(cid)
        if not c:
            print(f"\n  no such case: {cid}")
            continue
        print(f"\n{'=' * 70}\n{cid} [{c['difficulty']}] {c['question']}\n{'=' * 70}")
        print(c["gold_sql"].strip())
        cur = conn.execute(c["gold_sql"])
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        print(f"\n{' | '.join(cols)}")
        for r in rows[: args.rows]:
            print(" | ".join("NULL" if v is None else str(v) for v in r))
        if len(rows) > args.rows:
            print(f"... {len(rows)} rows total")

    if failures:
        return 1
    print("\nAll gold queries execute. Now eyeball the flagged ones with --show.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())