"""One-off aggregator: gpt-4o vs Claude multi-seed comparison.

Loads metrics from runs/run_001 (gpt-4o ab421), runs/run_016..024
(gpt-4o ab422..430), runs/run_005 (Claude ab421), and runs/run_007..015
(Claude ab422..430). Prints per-seed tables, aggregates, paired
comparison, and Wilcoxon signed-rank test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent.parent


def collect(provider_name: str, run_pairs):
    records = []
    for name, d in run_pairs:
        m = pd.read_csv(ROOT / d / "stage_5_eval" / "metrics.csv")
        cop = m[m["method"] == "copeland"].set_index("metric")["value"]
        records.append({
            "seed_ab": name,
            "tau_b": cop["tau_b"],
            "spearman": cop["spearman_rho"],
            "pair_acc": cop["pairwise_accuracy"],
        })
    df = pd.DataFrame(records)
    df["provider"] = provider_name
    return df


def aggregate(df):
    out = {}
    for col in ["tau_b", "spearman", "pair_acc"]:
        v = df[col].values
        n = len(v)
        mean = v.mean()
        std = v.std(ddof=1)
        se = std / np.sqrt(n)
        out[col] = {
            "mean": mean, "std": std, "se": se,
            "ci_lo": mean - 1.96 * se, "ci_hi": mean + 1.96 * se,
            "min": v.min(), "max": v.max(),
        }
    return out


def main():
    gpt_runs = [("ab421", "runs/run_001_seed_42")]
    for seed in range(422, 431):
        idx = seed - 406
        gpt_runs.append((f"ab{seed}", f"runs/run_{idx:03d}_seed_42_gpt4o_ab{seed}"))

    claude_runs = [("ab421", "runs/run_005_seed_42_claude")]
    for seed in range(422, 431):
        idx = seed - 415
        claude_runs.append((f"ab{seed}", f"runs/run_{idx:03d}_seed_42_claude_ab{seed}"))

    df_gpt = collect("gpt-4o", gpt_runs)
    df_claude = collect("Claude Sonnet 4.6", claude_runs)

    print("=== gpt-4o per-seed ===")
    print(df_gpt[["seed_ab", "tau_b", "spearman", "pair_acc"]].round(4).to_string(index=False))
    print()
    print("=== Claude per-seed ===")
    print(df_claude[["seed_ab", "tau_b", "spearman", "pair_acc"]].round(4).to_string(index=False))
    print()

    agg_gpt = aggregate(df_gpt)
    agg_claude = aggregate(df_claude)

    print("=" * 90)
    print(f'{"":>22} | {"gpt-4o (n=10)":>30} | {"Claude Sonnet 4.6 (n=10)":>30}')
    print("-" * 90)
    for col in ["tau_b", "spearman", "pair_acc"]:
        g = agg_gpt[col]
        c = agg_claude[col]
        print(f'{col:>15} mean | {g["mean"]:>+30.4f} | {c["mean"]:>+30.4f}')
        print(f'{"":>15}  std | {g["std"]:>30.4f} | {c["std"]:>30.4f}')
        print(f'{"":>15}   SE | {g["se"]:>30.4f} | {c["se"]:>30.4f}')
        print(f'{"":>15} CI95 | [{g["ci_lo"]:+.4f}, {g["ci_hi"]:+.4f}]              | [{c["ci_lo"]:+.4f}, {c["ci_hi"]:+.4f}]')
        print(f'{"":>15} range| [{g["min"]:+.4f}, {g["max"]:+.4f}]               | [{c["min"]:+.4f}, {c["max"]:+.4f}]')
        print()

    print("=== Paired comparison (Claude minus gpt-4o, same seed) ===")
    merged = df_gpt[["seed_ab", "tau_b"]].rename(columns={"tau_b": "gpt4o"}).merge(
        df_claude[["seed_ab", "tau_b"]].rename(columns={"tau_b": "claude"}),
        on="seed_ab",
    )
    merged["delta"] = merged["claude"] - merged["gpt4o"]
    print(merged.round(4).to_string(index=False))
    print()

    diffs = merged["delta"].values
    n_pos = int((diffs > 0).sum())
    n_zero = int((diffs == 0).sum())
    n_neg = int((diffs < 0).sum())
    print(f'mean(Claude - gpt-4o): {diffs.mean():+.4f}')
    print(f'std(diff): {diffs.std(ddof=1):.4f}')
    print(f'SE(diff): {diffs.std(ddof=1) / np.sqrt(len(diffs)):.4f}')
    print(f'sign: {n_pos} Claude>gpt-4o, {n_zero} ties, {n_neg} gpt-4o>Claude (out of 10)')

    res = wilcoxon(merged["claude"], merged["gpt4o"], alternative="greater")
    print(f'Wilcoxon signed-rank (H1: Claude > gpt-4o): stat={res.statistic:.2f}, p={res.pvalue:.4f}')


if __name__ == "__main__":
    main()
