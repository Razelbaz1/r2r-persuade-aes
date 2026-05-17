"""PERSUADE/LLM Phase-1 -- Predictions distribution review (script form).

Mirror of `01_predictions_distribution.ipynb` for users who can't get
VS Code Jupyter to start a kernel. Same outputs:
  * basic stats on predictions.csv
  * QWK reproduction check vs datafan07's published 0.83
  * Figure 1: true_score distribution
  * Figure 2A: predicted_score histogram (float)
  * Figure 2B: predicted_score rounded value_counts
  * Figure 3: top-10 overlay on the prediction histogram
  * Top-10 + top-50 tables (top-10 rows marked with *** at the start)

PNG figures land in `figures/` next to this script, 300 dpi.
Run from anywhere -- paths resolve relative to this file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive; saves PNGs without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# === Paths ===
SCRIPT_DIR = Path(__file__).resolve().parent
PREDICTIONS_PATH = (SCRIPT_DIR / ".." / "runs" / "run_001_seed_42" / "stage_1_aes" / "predictions.csv").resolve()
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    # === Load ===
    section("LOAD")
    df = pd.read_csv(PREDICTIONS_PATH)
    print(f"Loaded: {PREDICTIONS_PATH}")
    print(f"Shape:  {df.shape}")
    print(f"Cols:   {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes.to_string()}")
    print(
        f"\ntrue_score      range:  [{df['true_score'].min()}, {df['true_score'].max()}]  (int)"
    )
    print(
        f"predicted_score range:  [{df['predicted_score'].min():.4f}, "
        f"{df['predicted_score'].max():.4f}]  (float)"
    )

    # === Reproduction check vs datafan07 published CV ===
    section("REPRODUCTION CHECK vs datafan07 published CV")
    y_pred_int = df["predicted_score"].round().clip(1, 6).astype(int)
    y_true = df["true_score"]
    our_qwk = cohen_kappa_score(y_true, y_pred_int, weights="quadratic")
    datafan07_published = 0.83
    print(f"Our reproduction QWK:    {our_qwk:.4f}")
    print(f"datafan07 published CV:  {datafan07_published:.4f}")
    print(f"Difference:              {(our_qwk - datafan07_published):+.4f}")
    print("\nConfusion matrix (true rows x predicted cols):")
    cm = pd.crosstab(y_true, y_pred_int, rownames=["true"], colnames=["pred"], margins=True)
    print(cm.to_string())

    # === Figure 1: True score distribution ===
    section("FIGURE 1: True score distribution")
    true_counts = df["true_score"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(true_counts.index, true_counts.values, edgecolor="black", color="steelblue")
    for x, v in zip(true_counts.index, true_counts.values):
        ax.text(x, v + 50, f"{v:,}", ha="center", fontsize=10)
    ax.set_xlabel("True Score (human grader)")
    ax.set_ylabel("Number of essays")
    ax.set_title(f"Figure 1 -- True score distribution (PERSUADE 2.0, n={len(df):,})")
    ax.set_xticks(true_counts.index)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig1_true_score_distribution.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved -> {fig_path}")
    print("Counts:")
    print(true_counts.to_string())

    # === Figure 2A: Predicted score histogram (float) ===
    section("FIGURE 2A: Predicted score distribution (float, 50 bins)")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["predicted_score"], bins=50, edgecolor="black", color="darkorange", alpha=0.85)
    ax.set_xlabel("Predicted Score (LightGBM OOF, float)")
    ax.set_ylabel("Number of essays")
    ax.set_title(f"Figure 2A -- Predicted score (float, 50 bins, n={len(df):,})")
    ax.axvline(
        df["predicted_score"].mean(),
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"mean = {df['predicted_score'].mean():.3f}",
    )
    ax.axvline(
        df["predicted_score"].median(),
        color="blue",
        linestyle="--",
        linewidth=1,
        label=f"median = {df['predicted_score'].median():.3f}",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig2a_predicted_score_float.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved -> {fig_path}")
    print("predicted_score statistics:")
    print(df["predicted_score"].describe().to_string())

    # === Figure 2B: Predicted score rounded value_counts ===
    section("FIGURE 2B: Predicted score (rounded value_counts)")
    df["pred_rounded"] = df["predicted_score"].round().clip(1, 6).astype(int)
    pred_counts = df["pred_rounded"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(pred_counts.index, pred_counts.values, edgecolor="black", color="darkorange")
    for x, v in zip(pred_counts.index, pred_counts.values):
        ax.text(x, v + 50, f"{v:,}", ha="center", fontsize=10)
    ax.set_xlabel("Predicted Score (rounded, clipped to [1, 6])")
    ax.set_ylabel("Number of essays")
    ax.set_title(f"Figure 2B -- Predicted score (rounded, n={len(df):,})")
    ax.set_xticks(range(1, 7))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig2b_predicted_score_rounded.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved -> {fig_path}")
    print("True vs predicted (rounded) side by side:")
    comparison = pd.DataFrame(
        {
            "true (Fig 1)": true_counts.reindex(range(1, 7), fill_value=0),
            "predicted (Fig 2B)": pred_counts.reindex(range(1, 7), fill_value=0),
        }
    )
    comparison["diff (pred - true)"] = (
        comparison["predicted (Fig 2B)"] - comparison["true (Fig 1)"]
    )
    print(comparison.to_string())

    # === Top-10 ===
    section("TOP-10 essays (by predicted_score)")
    top_10 = df.sort_values("predicted_score", ascending=False).head(10).reset_index(drop=True)
    top_10.insert(0, "rank", range(1, 11))
    view_10 = top_10[["rank", "essay_id", "predicted_score", "true_score"]].copy()
    view_10["predicted_score"] = view_10["predicted_score"].round(4)
    print(view_10.to_string(index=False))
    print()
    print(f"Mean true_score in top-10:               {top_10['true_score'].mean():.2f}")
    print(f"# of top-10 with true_score == 6:        {(top_10['true_score'] == 6).sum()} / 10")
    print(f"# of top-10 with true_score >= 5:        {(top_10['true_score'] >= 5).sum()} / 10")

    # === Figure 3: Top-10 overlay ===
    section("FIGURE 3: Top-10 overlay on prediction distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        df["predicted_score"],
        bins=50,
        edgecolor="black",
        color="lightgray",
        alpha=0.7,
        label=f"All essays (n={len(df):,})",
    )
    for _, row in top_10.iterrows():
        ax.axvline(row["predicted_score"], color="red", linestyle="-", linewidth=1.5, alpha=0.8)
    ax.axvline(
        top_10["predicted_score"].iloc[-1],
        color="red",
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
        label="Top-10 essays",
    )
    ax.set_xlabel("Predicted Score")
    ax.set_ylabel("Number of essays")
    ax.set_title("Figure 3 -- Top-10 essays (red lines) on the prediction distribution")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig3_top10_on_distribution.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved -> {fig_path}")

    # === Top-50 with top-10 marked ===
    section("TOP-50 (top-10 marked with ***)")
    top_50 = df.sort_values("predicted_score", ascending=False).head(50).reset_index(drop=True)
    top_50.insert(0, "rank", range(1, 51))
    top_50["in_top_10"] = top_50["rank"] <= 10
    top_50["marker"] = top_50["in_top_10"].map({True: "*** ", False: "    "})
    view_50 = top_50[["marker", "rank", "essay_id", "predicted_score", "true_score"]].copy()
    view_50["predicted_score"] = view_50["predicted_score"].round(4)
    print(view_50.to_string(index=False))
    print()
    print("Distribution of true_score in top-50:")
    print(top_50["true_score"].value_counts().sort_index().to_string())
    print()
    print(f"Mean true_score in top-50:               {top_50['true_score'].mean():.2f}")
    print(f"# true=6 in top-10: {((top_50['rank'] <= 10) & (top_50['true_score'] == 6)).sum()} / 10")
    print(f"# true=6 in top-50: {(top_50['true_score'] == 6).sum()} / 50")
    print(f"# true=5 in top-50: {(top_50['true_score'] == 5).sum()} / 50")
    print(f"# true=4 in top-50: {(top_50['true_score'] == 4).sum()} / 50")
    print(f"# true<4 in top-50: {(top_50['true_score'] < 4).sum()} / 50")

    section("DONE")
    print(f"All figures saved under: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
