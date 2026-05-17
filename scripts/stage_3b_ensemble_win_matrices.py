"""PS3b -- ensemble two or more PS3 win matrices into one.

For Phase-2 fusion. Each source run-dir must contain a
`stage_3_llm/win_matrix.csv` with identical essay_id index/columns.
The output is the element-wise sum, written to the target run-dir's
`stage_3_llm/win_matrix.csv`. Cell values are no longer binary; ties
appear at `(W[i,j], W[j,i]) = (1, 1)` whenever the source judges
disagreed on a pair. PS4 already supports this via `COPELAND_ALPHA`.

Also concatenates the source `pairwise_log.csv` files into a combined
log (per pair, one row per source judge), so the audit trail survives
the ensemble step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import add_package_to_syspath, resolve_or_create_run_dir, stage_dir

add_package_to_syspath()


def main(target_dir: Path, source_dirs: list[Path]) -> tuple[Path, Path]:
    if len(source_dirs) < 2:
        raise ValueError(
            f"At least two source runs required for an ensemble; got {len(source_dirs)}."
        )

    matrices = []
    logs = []
    for src in source_dirs:
        wm_path = src / "stage_3_llm" / "win_matrix.csv"
        if not wm_path.exists():
            raise FileNotFoundError(f"Source missing win_matrix.csv: {wm_path}")
        W = pd.read_csv(wm_path, index_col=0)
        W.index = W.index.astype(str)
        W.columns = W.columns.astype(str)
        matrices.append((src.name, W))

        log_path = src / "stage_3_llm" / "pairwise_log.csv"
        if log_path.exists():
            log = pd.read_csv(log_path)
            log["source_run"] = src.name
            logs.append(log)

    # Verify identical index/columns across sources
    ref_name, ref = matrices[0]
    for name, W in matrices[1:]:
        if list(W.index) != list(ref.index):
            raise ValueError(
                f"Index mismatch between {ref_name} and {name}: "
                f"different essay_id ordering or content."
            )
        if list(W.columns) != list(ref.columns):
            raise ValueError(
                f"Column mismatch between {ref_name} and {name}."
            )

    summed = ref.copy()
    for name, W in matrices[1:]:
        summed = summed + W

    out_dir = stage_dir(target_dir, "stage_3_llm")
    wm_out = out_dir / "win_matrix.csv"
    summed.to_csv(wm_out)

    log_out = out_dir / "pairwise_log.csv"
    if logs:
        merged_log = pd.concat(logs, ignore_index=True)
        merged_log.to_csv(log_out, index=False)

    n_pairs = (summed.values.sum() // 2) // len(matrices)
    n_ties = int(((summed == 1) & (summed.T == 1)).values.sum() // 2)
    print(
        f"PS3b done. Ensembled {len(matrices)} source matrices "
        f"({', '.join(name for name, _ in matrices)}) into {wm_out}. "
        f"Pairs: {n_pairs}. Disagreement ties (1, 1): {n_ties}."
    )
    return wm_out, log_out


def cli() -> None:
    parser = argparse.ArgumentParser(description="PS3b -- ensemble PS3 win matrices")
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument(
        "--source-runs",
        type=Path,
        nargs="+",
        required=True,
        help="Two or more source run-dirs to ensemble. Each must hold "
             "stage_3_llm/win_matrix.csv with identical essay_id index.",
    )
    args = parser.parse_args()
    target = resolve_or_create_run_dir(seed=-1, explicit=args.target_run)
    main(target_dir=target, source_dirs=args.source_runs)


if __name__ == "__main__":
    cli()
