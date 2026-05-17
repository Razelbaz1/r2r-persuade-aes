"""PS5 — Evaluate AES-only and Copeland rankings against true human scores.

Computes Kendall tau-b, Spearman rho, and pairwise accuracy on each
method's ranking over the top-k items. The top-k items come from
`<run-dir>/stage_2_top_k/top_k.csv`; the AES predicted score is the
ranking source for the `aes` method, the Copeland score is the ranking
source for the `copeland` method.

Pairwise accuracy skips pairs with equal true scores (Lihi playbook
step 15). All three metrics are computed only over the top-k subset.

Writes `<run-dir>/stage_5_eval/metrics.csv` with columns
(method, metric, value) — long format so adding methods/metrics in
Phase 2 is append-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import add_package_to_syspath, resolve_or_create_run_dir, stage_dir

add_package_to_syspath()
from lib.metrics import evaluate_ranking  # noqa: E402


METHODS = ("aes", "copeland")


def main(run_dir: Path) -> Path:
    top_k_path = run_dir / "stage_2_top_k" / "top_k.csv"
    copeland_path = run_dir / "stage_4_fusion" / "copeland_scores.csv"
    if not top_k_path.exists():
        raise FileNotFoundError(f"Missing PS2 output: {top_k_path}")
    if not copeland_path.exists():
        raise FileNotFoundError(f"Missing PS4 output: {copeland_path}")

    top_k = pd.read_csv(top_k_path)
    copeland = pd.read_csv(copeland_path)
    if set(top_k["essay_id"]) != set(copeland["essay_id"]):
        raise ValueError(
            "essay_id sets differ between top_k.csv and copeland_scores.csv"
        )

    true_scores = pd.Series(
        top_k["true_score"].values, index=top_k["essay_id"].values, name="true"
    )
    aes_scores = pd.Series(
        top_k["predicted_score"].values, index=top_k["essay_id"].values, name="aes"
    )
    copeland_scores_series = pd.Series(
        copeland["copeland_score"].values,
        index=copeland["essay_id"].values,
        name="copeland",
    )

    rows = []
    for method, pred in (("aes", aes_scores), ("copeland", copeland_scores_series)):
        metrics = evaluate_ranking(pred_scores_by_item=pred, true_scores_by_item=true_scores)
        for metric_name, value in metrics.items():
            rows.append({"method": method, "metric": metric_name, "value": value})

    out_dir = stage_dir(run_dir, "stage_5_eval")
    out_path = out_dir / "metrics.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def cli() -> None:
    parser = argparse.ArgumentParser(description="PS5 — evaluate rankings")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = resolve_or_create_run_dir(seed=-1, explicit=args.run_dir)
    out_path = main(run_dir=run_dir)
    print(f"PS5 done. Wrote {out_path}")


if __name__ == "__main__":
    cli()
