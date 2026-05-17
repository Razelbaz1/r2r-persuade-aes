"""PS2 -- Select a window of essays from the AES baseline predictions.

Reads `<run-dir>/stage_1_aes/predictions.csv` (must contain at least
columns `essay_id`, `predicted_score`, `true_score`, `essay_text`),
sorts by `predicted_score` descending, and writes the selected window
of rows to `<run-dir>/stage_2_top_k/top_k.csv` with a
`rank_position` column starting at 1 (relative to the selection).
An extra `absolute_rank` column is added to preserve the row's
original AES rank (1 = global best predicted essay).

Two selection modes (mutually exclusive):

* `--top-k N` (default): ranks 1..N -- the highest predicted essays.
* `--rank-from A --rank-to B`: ranks A..B inclusive (1-indexed) -- a
   window further down the ranking. Useful when the AES is already
   near-perfect at the top and we want to test pairwise refinement
   on a "noisy" region where predicted_score differences are tiny.

Tie handling is stable on the original order from PS1 (mergesort).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import resolve_or_create_run_dir, stage_dir


def main(
    run_dir: Path,
    rank_from: int = 1,
    rank_to: int = 10,
) -> Path:
    """Select essays whose AES rank is in the inclusive range [rank_from, rank_to].

    Both bounds are 1-indexed. Rank 1 is the essay with the highest
    `predicted_score`.
    """
    if rank_from < 1:
        raise ValueError(f"rank_from must be >= 1, got {rank_from}")
    if rank_to < rank_from:
        raise ValueError(f"rank_to ({rank_to}) must be >= rank_from ({rank_from})")

    predictions_path = run_dir / "stage_1_aes" / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Expected AES predictions at {predictions_path}. Run PS1 first."
        )
    predictions = pd.read_csv(predictions_path)
    required = {"essay_id", "predicted_score", "true_score", "essay_text"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions.csv missing columns: {missing}")

    # Sort by predicted_score descending and assign absolute rank.
    sorted_df = (
        predictions.sort_values("predicted_score", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    if rank_to > len(sorted_df):
        raise ValueError(
            f"rank_to ({rank_to}) exceeds the number of essays in predictions.csv "
            f"({len(sorted_df)})."
        )
    sorted_df.insert(0, "absolute_rank", range(1, len(sorted_df) + 1))

    # iloc is 0-based; rank_from / rank_to are 1-based inclusive.
    window = sorted_df.iloc[rank_from - 1 : rank_to].reset_index(drop=True)
    window.insert(0, "rank_position", range(1, len(window) + 1))

    out_dir = stage_dir(run_dir, "stage_2_top_k")
    out_path = out_dir / "top_k.csv"
    window.to_csv(out_path, index=False)
    return out_path


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="PS2 -- select a window of essays by AES rank"
    )
    parser.add_argument("--run-dir", type=Path, required=True)

    # Mutually exclusive selection modes.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Convenience: select ranks 1..N. Mutually exclusive with --rank-from/--rank-to.",
    )
    # Range flags are NOT in the mutually-exclusive group so they can be
    # used together; we just ban combining them with --top-k.
    parser.add_argument(
        "--rank-from",
        type=int,
        default=None,
        help="Lower bound of the AES rank window (1-indexed, inclusive). Default 1.",
    )
    parser.add_argument(
        "--rank-to",
        type=int,
        default=None,
        help="Upper bound of the AES rank window (1-indexed, inclusive). Default 10.",
    )

    args = parser.parse_args()

    if args.top_k is not None and (args.rank_from is not None or args.rank_to is not None):
        parser.error("--top-k cannot be combined with --rank-from / --rank-to.")

    if args.top_k is not None:
        rank_from, rank_to = 1, args.top_k
    else:
        rank_from = args.rank_from if args.rank_from is not None else 1
        rank_to = args.rank_to if args.rank_to is not None else 10

    run_dir = resolve_or_create_run_dir(seed=-1, explicit=args.run_dir)
    out_path = main(run_dir=run_dir, rank_from=rank_from, rank_to=rank_to)
    print(
        f"PS2 done. Selected ranks {rank_from}..{rank_to} "
        f"({rank_to - rank_from + 1} essays). Wrote {out_path}"
    )


if __name__ == "__main__":
    cli()
