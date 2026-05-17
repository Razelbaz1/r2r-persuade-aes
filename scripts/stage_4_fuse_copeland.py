"""PS4 — Copeland fusion of the LLM win matrix.

Reads `<run-dir>/stage_3_llm/win_matrix.csv` (square, indexed by
essay_id) and produces two outputs in
`<run-dir>/stage_4_fusion/`:

- `copeland_scores.csv` — per-essay Copeland score
- `ranking_copeland.csv` — essays ranked by Copeland score (descending),
  with a `rank_position` column

Tie credit (`COPELAND_ALPHA`) is held in this script so it travels with
the run; it has no effect in Phase 1 because each pair carries a single
binary LLM vote (no ties possible). It is wired in for Phase 2 where
multiple LLM calls per pair may produce ties.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import add_package_to_syspath, resolve_or_create_run_dir, stage_dir

add_package_to_syspath()
from lib.voting import copeland_scores, order_by_score_desc  # noqa: E402

COPELAND_ALPHA = 1.0 / 3.0  # Matches `pipeline/run_pipeline.py:65`.


def main(run_dir: Path) -> tuple[Path, Path]:
    win_matrix_path = run_dir / "stage_3_llm" / "win_matrix.csv"
    if not win_matrix_path.exists():
        raise FileNotFoundError(
            f"Expected win matrix at {win_matrix_path}. Run PS3 first."
        )
    W = pd.read_csv(win_matrix_path, index_col=0)
    # PERSUADE essay_id is a hex string (e.g. "1b1975b"); keep it as str
    # rather than forcing int conversion (which crashes on hex).
    W.index = W.index.astype(str)
    W.columns = W.columns.astype(str)
    # Square + symmetric-index sanity check.
    if list(W.index) != list(W.columns):
        raise ValueError("win_matrix.csv must have identical row and column index.")

    scores = copeland_scores(W, alpha=COPELAND_ALPHA)
    ordered = order_by_score_desc(scores)

    out_dir = stage_dir(run_dir, "stage_4_fusion")
    scores_path = out_dir / "copeland_scores.csv"
    pd.DataFrame(
        {"essay_id": scores.index, "copeland_score": scores.values}
    ).to_csv(scores_path, index=False)

    ranking_path = out_dir / "ranking_copeland.csv"
    pd.DataFrame(
        {
            "rank_position": range(1, len(ordered) + 1),
            "essay_id": ordered,
            "copeland_score": [float(scores.loc[i]) for i in ordered],
        }
    ).to_csv(ranking_path, index=False)

    return scores_path, ranking_path


def cli() -> None:
    parser = argparse.ArgumentParser(description="PS4 — Copeland fusion")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = resolve_or_create_run_dir(seed=-1, explicit=args.run_dir)
    scores_path, ranking_path = main(run_dir=run_dir)
    print(f"PS4 done. Wrote {scores_path} and {ranking_path}")


if __name__ == "__main__":
    cli()
