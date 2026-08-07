"""Decide whether a generated query's results match the gold query's.

This module defines execution accuracy, which makes it the most consequential
file in the evaluation path: if comparison is wrong, every number in your
report is wrong, in a way that looks perfectly plausible. It is also the one
module in the project genuinely worth unit-testing.

The rules, and why each exists:

**Compare results, not SQL text.** Many different queries are correct. Text
similarity would punish a model for writing a valid join differently.

**Rows are an unordered multiset unless the question implies order.** "How many
orders per status" has no natural ordering, so demanding one creates false
failures. "Top 5 sellers" does, so ignoring it would let a wrongly-sorted
answer pass. The ``ordered`` flag in testset.yaml decides per case.

**Duplicates matter.** A multiset, not a set: a query returning a row twice is
not equivalent to one returning it once, and collapsing to a set would hide
a missing GROUP BY.

**Columns compare by position, not name.** Aliases legitimately differ --
``AS revenue`` versus ``AS total_revenue`` is not an error. Optionally a
column permutation is also accepted; see ``allow_permutation``.

**Floats compare with tolerance.** ``SUM(price)`` accumulated in a different
join order can differ in the last bits. Exact equality would fail correct
queries at random.

What this module deliberately does *not* do is decide whether the gold query
answers its question. Only a human can do that, which is what
``validate_testset.py --show`` is for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Sequence

Row = Sequence[Any]

#: Absolute/relative tolerance for float comparison. Tight enough that a real
#: arithmetic error still fails, loose enough to absorb accumulation noise.
DEFAULT_TOL = 1e-6

#: A looser bar used only for diagnostics: a failure that would pass at this
#: tolerance is almost always the model applying ROUND() when the gold query
#: did not. Reported separately -- never counted as a pass, because silently
#: widening the metric to flatter yourself is how evaluations lose meaning.
NEAR_MISS_TOL = 0.005

#: Guard rails for the permutation search, which is factorial in column count.
MAX_PERMUTATION_COLUMNS = 5
MAX_PERMUTATION_ROWS = 5000


@dataclass
class ComparisonResult:
    """Outcome of comparing one predicted result set against its gold."""

    match: bool
    #: "exact" -- matched with columns in gold order.
    #: "permuted" -- matched only after reordering predicted columns.
    #: "mismatch" -- did not match.
    kind: str = "mismatch"
    #: Human-readable cause, written to be read in a failure log. This is the
    #: field you will actually use when deciding what to improve in Step 4.
    reason: str | None = None
    #: Both result sets were empty. Counted as a match, but flagged: an empty
    #: gold result cannot distinguish a correct query from a broken one, so
    #: these should be reviewed rather than trusted.
    both_empty: bool = False
    #: Would have matched at NEAR_MISS_TOL. Signals a rounding difference.
    near_miss: bool = False
    #: Rows matched as a multiset but in a different order, on a case that
    #: required order. Tells you the model got the aggregation right and the
    #: sort wrong -- a much easier fix than it appears from a bare "failed".
    order_only: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.match:
            tag = "MATCH" + ("*" if self.kind == "permuted" else "")
            return f"{tag} ({self.kind})" + (" [both empty]" if self.both_empty else "")
        extra = []
        if self.order_only:
            extra.append("order only")
        if self.near_miss:
            extra.append("rounding")
        suffix = f" [{', '.join(extra)}]" if extra else ""
        return f"MISMATCH: {self.reason}{suffix}"


# ---------------------------------------------------------------------------
# Value-level comparison
# ---------------------------------------------------------------------------

def _is_number(v: Any) -> bool:
    # bool is a subclass of int; treated as numeric so 1 == True, which is
    # what SQLite's own type affinity would give you anyway.
    return isinstance(v, (int, float, bool)) and not isinstance(v, str)


def values_equal(a: Any, b: Any, tol: float = DEFAULT_TOL) -> bool:
    """Compare two cell values.

    NULL equals NULL. Numbers compare with tolerance and across int/float, so
    ``5`` and ``5.0`` match. Strings are stripped before comparison, because
    trailing whitespace in source data is not a modelling error. Everything
    else falls back to string equality.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    if _is_number(a) and _is_number(b):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        if math.isinf(fa) or math.isinf(fb):
            return fa == fb
        return math.isclose(fa, fb, rel_tol=tol, abs_tol=tol)

    # A number against its string form ("5" vs 5) counts as equal: which one
    # you get depends on CAST choices that do not change the answer.
    if _is_number(a) != _is_number(b):
        try:
            return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
        except (TypeError, ValueError):
            return False

    return str(a).strip() == str(b).strip()


def _sort_key(row: Row, tol: float = DEFAULT_TOL) -> tuple:
    """Canonical key for ordering rows before an unordered comparison.

    Python cannot sort mixed types, so each value becomes a
    ``(type_rank, value)`` pair. Numbers are rounded into buckets sized by the
    tolerance, so two values that compare equal almost always sort adjacently.

    Known limitation: two values straddling a bucket boundary (1.9999995 and
    2.0000005) can sort apart and be reported as a mismatch. Vanishingly rare
    with real aggregates, but worth knowing before you chase a phantom bug.
    """
    ndigits = max(0, int(round(-math.log10(tol)))) if tol > 0 else 12
    key = []
    for v in row:
        if v is None:
            key.append((0, 0.0, ""))
        elif _is_number(v):
            f = float(v)
            key.append((1, round(f, ndigits) if math.isfinite(f) else f, ""))
        else:
            key.append((2, 0.0, str(v).strip()))
    return tuple(key)


def _rows_equal_positional(
    gold: Sequence[Row], pred: Sequence[Row], tol: float
) -> tuple[bool, str | None]:
    """Element-wise comparison of two equally-sized row lists."""
    for i, (g, p) in enumerate(zip(gold, pred)):
        if len(g) != len(p):
            return False, f"row {i}: column count differs ({len(g)} vs {len(p)})"
        for j, (gv, pv) in enumerate(zip(g, p)):
            if not values_equal(gv, pv, tol):
                return False, f"row {i}, column {j}: expected {gv!r}, got {pv!r}"
    return True, None


def rows_match(
    gold: Sequence[Row],
    pred: Sequence[Row],
    ordered: bool = False,
    tol: float = DEFAULT_TOL,
) -> tuple[bool, str | None]:
    """Compare two row lists, honouring the ordered flag."""
    if len(gold) != len(pred):
        return False, f"row count differs (expected {len(gold)}, got {len(pred)})"
    if not gold:
        return True, None

    if ordered:
        return _rows_equal_positional(gold, pred, tol)

    g_sorted = sorted(gold, key=lambda r: _sort_key(r, tol))
    p_sorted = sorted(pred, key=lambda r: _sort_key(r, tol))
    return _rows_equal_positional(g_sorted, p_sorted, tol)


# ---------------------------------------------------------------------------
# Result-set comparison
# ---------------------------------------------------------------------------

def _permuted_match(
    gold: Sequence[Row], pred: Sequence[Row], ordered: bool, tol: float
) -> list[int] | None:
    """Try every column ordering of ``pred``; return the one that matches.

    Selecting ``revenue, category`` instead of ``category, revenue`` answers
    the question equally well, and failing it measures formatting compliance
    rather than SQL skill. Bounded by column and row count because the search
    is factorial.
    """
    ncols = len(gold[0]) if gold else 0
    if ncols == 0 or ncols > MAX_PERMUTATION_COLUMNS or len(gold) > MAX_PERMUTATION_ROWS:
        return None

    for perm in permutations(range(ncols)):
        if perm == tuple(range(ncols)):
            continue  # already tried as the exact comparison
        reordered = [tuple(row[i] for i in perm) for row in pred]
        ok, _ = rows_match(gold, reordered, ordered=ordered, tol=tol)
        if ok:
            return list(perm)
    return None


def compare_results(
    gold_columns: Sequence[str] | None,
    gold_rows: Sequence[Row],
    pred_columns: Sequence[str] | None,
    pred_rows: Sequence[Row],
    ordered: bool = False,
    tol: float = DEFAULT_TOL,
    allow_permutation: bool = True,
) -> ComparisonResult:
    """Compare a predicted result set against the gold one.

    Args:
        gold_columns / gold_rows: Result of the gold query.
        pred_columns / pred_rows: Result of the agent's query.
        ordered: True when row order is part of correctness (top-N questions).
        tol: Float comparison tolerance.
        allow_permutation: Accept a match after reordering predicted columns.
            Set False for a stricter metric; the result's ``kind`` field lets
            you report both numbers from a single run either way.

    Column *names* are ignored entirely -- only shape and values are compared.
    """
    gold_rows = [tuple(r) for r in gold_rows]
    pred_rows = [tuple(r) for r in pred_rows]

    if not gold_rows and not pred_rows:
        return ComparisonResult(
            match=True, kind="exact", both_empty=True,
            reason="both result sets empty",
        )

    if not gold_rows:
        return ComparisonResult(
            match=False,
            reason=f"expected no rows, got {len(pred_rows)}",
        )
    if not pred_rows:
        return ComparisonResult(
            match=False,
            reason=f"expected {len(gold_rows)} row(s), got none",
        )

    n_gold_cols, n_pred_cols = len(gold_rows[0]), len(pred_rows[0])
    if n_gold_cols != n_pred_cols:
        # Usually SELECT * against a gold query that names its columns.
        return ComparisonResult(
            match=False,
            reason=f"column count differs (expected {n_gold_cols}, got {n_pred_cols})",
            details={"gold_columns": list(gold_columns or []),
                     "pred_columns": list(pred_columns or [])},
        )

    ok, reason = rows_match(gold_rows, pred_rows, ordered=ordered, tol=tol)
    if ok:
        return ComparisonResult(match=True, kind="exact")

    # --- diagnostics: work out *why*, since that drives Step 4 decisions ---
    result = ComparisonResult(match=False, reason=reason)

    if ordered:
        unordered_ok, _ = rows_match(gold_rows, pred_rows, ordered=False, tol=tol)
        if unordered_ok:
            result.order_only = True
            result.reason = "correct rows, wrong order"
            return result

    near_ok, _ = rows_match(gold_rows, pred_rows, ordered=ordered, tol=NEAR_MISS_TOL)
    if near_ok:
        result.near_miss = True
        result.reason = f"{reason} (would match at tolerance {NEAR_MISS_TOL})"
        return result

    if allow_permutation:
        perm = _permuted_match(gold_rows, pred_rows, ordered, tol)
        if perm is not None:
            return ComparisonResult(
                match=True, kind="permuted",
                reason="matched after reordering predicted columns",
                details={"permutation": perm},
            )

    return result


def compare_case(
    case: dict[str, Any],
    gold_columns: Sequence[str] | None,
    gold_rows: Sequence[Row],
    pred_columns: Sequence[str] | None,
    pred_rows: Sequence[Row],
    **kwargs: Any,
) -> ComparisonResult:
    """Compare using a test case's own ``ordered`` flag.

    The convenience wrapper the harness actually calls -- it keeps the
    ordered-flag lookup in one place rather than at every call site.
    """
    return compare_results(
        gold_columns, gold_rows, pred_columns, pred_rows,
        ordered=bool(case.get("ordered", False)),
        **kwargs,
    )