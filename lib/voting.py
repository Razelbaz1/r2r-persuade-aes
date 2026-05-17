"""Voting-rule aggregation for the LLM pairwise refinement experiment.

Copeland scoring is the Phase-1 fusion rule. The implementation is the
same as the Sushi pipeline's
`pipeline/notebooks/_build_stage_3_notebooks.py:177-203`, except the
input here is a pure 0/1 win matrix (the LLM returns binary A/B per
playbook step 9) rather than a per-bucket pairwise count matrix. The
`alpha` (tie-credit) parameter is preserved for forward compatibility
with Phase 2, where multiple LLM calls per pair may produce ties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def copeland_scores(W: pd.DataFrame, alpha: float = 1.0 / 3.0) -> pd.Series:
    """Per-item Copeland score from a square win matrix.

    Parameters
    ----------
    W
        Square win matrix indexed and columned by item_id. `W.loc[i, j]`
        is the number of times i was preferred over j across LLM calls
        (in Phase 1 always 0 or 1; in Phase 2 up to N).
    alpha
        Tie credit, in [0, 1]. The Sushi pipeline locks `alpha = 1/3`
        (per `pipeline/run_pipeline.py:65`, `COPELAND_ALPHA`). Ties
        cannot occur in Phase 1 because each pair has a single binary
        vote, so this default has no effect; it matters in Phase 2.

    Returns
    -------
    pd.Series indexed by item_id, named `"copeland"`.
    """
    items = list(W.index)
    M = W.values.astype(float)
    n = len(items)
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            wij = M[i, j]
            wji = M[j, i]
            if np.isnan(wij) or np.isnan(wji):
                continue
            if wij > wji:
                scores[i] += 1.0
            elif wij == wji:
                scores[i] += alpha
    return pd.Series(scores, index=items, name="copeland")


def order_by_score_desc(scores: pd.Series) -> list:
    """Sort item_ids descending by score; stable tie-break by item_id ascending."""
    pairs = [(item, float(sc)) for item, sc in scores.items()]
    pairs.sort(key=lambda x: (-x[1], x[0]))
    return [p[0] for p in pairs]
