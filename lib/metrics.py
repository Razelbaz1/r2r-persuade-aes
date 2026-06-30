"""Ranking-evaluation metrics for the LLM pairwise refinement experiment.

The three metrics in Phase 1 follow Lihi's playbook (`for_raz.tex`,
steps 14-15): Kendall's tau-b, Spearman's rho, and pairwise accuracy.
All three are computed only inside the top-k subset against the true
human scores. Pairwise accuracy skips pairs with equal true scores per
step 15.

`partial_order_accuracy` (PACP) is the headline metric Lihi reports in
`SCALA_results.ipynb`: partial-order pairwise accuracy with half credit
for predicted ties, evaluated on tiers of equal true score. It differs
from `pairwise_accuracy` above only in tie handling (0.5 credit) and in
operating on explicit tiers rather than raw scores.

References to the Sushi implementations preserved for traceability:
- tau_b: `pipeline/notebooks/stage_3_per_user.ipynb` cell ~280
- pairwise accuracy: same notebook cell ~250 (PACP)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def kendall_tau_b(pred_scores: Sequence[float], true_scores: Sequence[float]) -> float:
    """Kendall's tau-b, tie-adjusted.

    Returns NaN if either series is constant (correlation undefined).
    """
    tau, _ = kendalltau(pred_scores, true_scores, variant="b")
    return float(tau) if tau is not None and not np.isnan(tau) else float("nan")


def spearman_rho(pred_scores: Sequence[float], true_scores: Sequence[float]) -> float:
    """Spearman's rank correlation coefficient.

    Returns NaN if either series is constant.
    """
    rho, _ = spearmanr(pred_scores, true_scores)
    return float(rho) if rho is not None and not np.isnan(rho) else float("nan")


def pairwise_accuracy(
    pred_scores: Sequence[float],
    true_scores: Sequence[float],
) -> float:
    """Pairwise accuracy on all unordered pairs.

    For every pair (i, j) with i < j and `true_scores[i] != true_scores[j]`,
    the prediction is "correct" iff `sign(pred_scores[i] - pred_scores[j])`
    matches `sign(true_scores[i] - true_scores[j])`. Pairs with equal
    true scores are skipped (Lihi playbook step 15).

    Returns
    -------
    Fraction of comparable pairs predicted correctly, in [0, 1]. NaN if
    no comparable pairs exist.
    """
    p = np.asarray(pred_scores, dtype=float)
    t = np.asarray(true_scores, dtype=float)
    if p.shape != t.shape or p.ndim != 1:
        raise ValueError("pred_scores and true_scores must be 1-D and same length")
    n = len(p)
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dt = t[i] - t[j]
            if dt == 0:
                continue
            dp = p[i] - p[j]
            if np.sign(dp) == np.sign(dt):
                correct += 1
            total += 1
    if total == 0:
        return float("nan")
    return correct / total


def tiers_by_value(
    items: Sequence,
    value_by_item: dict,
    descending: bool = True,
) -> list[list]:
    """Group items into ordered tiers of equal value (best tier first).

    Items sharing a value form one tier; tiers are sorted by value
    (descending by default, so the highest-scored items land in tier 0).
    Builds the `true_groups` / `pred_groups` arguments of
    `partial_order_accuracy` from a per-item value: true human score,
    AES predicted score, or Copeland win count.
    """
    by_value: dict = {}
    for item in items:
        by_value.setdefault(value_by_item[item], []).append(item)
    return [by_value[v] for v in sorted(by_value, reverse=descending)]


def partial_order_accuracy(
    true_groups: Sequence[Sequence],
    pred_groups: Sequence[Sequence],
) -> float:
    """Partial-order pairwise accuracy (PACP) — Lihi's headline metric.

    Counts every pair of items drawn from two *different* true tiers (the
    earlier tier should outrank the later one). A pair scores 1.0 when the
    prediction ranks them in the correct tier order, 0.5 when the
    prediction ties them in one tier, and 0.0 when reversed. Within-tier
    true pairs are skipped — their order is immaterial.

    Mirrors `partial_order_accuracy` in `SCALA_results.ipynb`; reproduces
    the reference values 0.1724 / 0.4138 / 0.4828 on that notebook's toy
    example. Returns NaN when there are no cross-tier pairs.
    """
    pred_group: dict = {}
    for group_idx, group in enumerate(pred_groups):
        for item in group:
            pred_group[item] = group_idx

    correct = 0.0
    total = 0
    for higher_idx in range(len(true_groups)):
        for lower_idx in range(higher_idx + 1, len(true_groups)):
            for h in true_groups[higher_idx]:
                for l in true_groups[lower_idx]:
                    total += 1
                    if pred_group[h] < pred_group[l]:
                        correct += 1.0
                    elif pred_group[h] == pred_group[l]:
                        correct += 0.5
    return correct / total if total else float("nan")


def evaluate_ranking(
    pred_scores_by_item: pd.Series,
    true_scores_by_item: pd.Series,
) -> dict[str, float]:
    """Compute the three Phase-1 metrics on an aligned pair of Series.

    Both Series must be indexed by `item_id`. The function reindexes
    `true_scores_by_item` onto `pred_scores_by_item.index` before
    computing — the prediction's item set defines the evaluation window
    (e.g. for top-k metrics, pass the top-k subset only).
    """
    aligned_true = true_scores_by_item.reindex(pred_scores_by_item.index)
    if aligned_true.isna().any():
        missing = aligned_true.index[aligned_true.isna()].tolist()
        raise ValueError(f"true_scores missing for items: {missing}")
    return {
        "tau_b": kendall_tau_b(pred_scores_by_item.values, aligned_true.values),
        "spearman_rho": spearman_rho(pred_scores_by_item.values, aligned_true.values),
        "pairwise_accuracy": pairwise_accuracy(
            pred_scores_by_item.values, aligned_true.values
        ),
    }
