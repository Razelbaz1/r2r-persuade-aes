"""Builds Figure 4 (rank ladder slopegraph) for the Claude run.

Mirror of the fig4 notebook cell, but pointing at run_005 (Claude
Sonnet 4.6, ab421) instead of run_001 (gpt-4o ab421). Saves two
versions: title-less for formal reporting, titled for slides.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

NB_DIR = Path(__file__).parent
ROOT = NB_DIR.parent
TOP_K_PATH = ROOT / "runs" / "run_001_seed_42" / "stage_2_top_k" / "top_k.csv"
CLAUDE_RANKING_PATH = ROOT / "runs" / "run_005_seed_42_claude" / "stage_4_fusion" / "ranking_copeland.csv"
FIG_DIR = NB_DIR / "figures"


def build(with_title: bool) -> Path:
    top_k = pd.read_csv(TOP_K_PATH)
    top_k["essay_id"] = top_k["essay_id"].astype(str)
    ranking_copeland = pd.read_csv(CLAUDE_RANKING_PATH)
    ranking_copeland["essay_id"] = ranking_copeland["essay_id"].astype(str)

    aes_rank = {row.essay_id: i + 1 for i, row in enumerate(top_k.itertuples(index=False))}
    copeland_rank = {row.essay_id: int(row.rank_position) for row in ranking_copeland.itertuples(index=False)}

    eids = list(aes_rank.keys())
    df_plot = pd.DataFrame({
        "essay_id": eids,
        "true_score": [int(top_k.loc[top_k["essay_id"] == e, "true_score"].iloc[0]) for e in eids],
        "aes_rank": [aes_rank[e] for e in eids],
        "copeland_rank": [copeland_rank[e] for e in eids],
    })

    score_to_band = {}
    for score in sorted(df_plot["true_score"].unique(), reverse=True):
        members = df_plot[df_plot["true_score"] == score].sort_values("copeland_rank")
        if len(members) == 0:
            continue
        start = len(score_to_band)
        for offset, eid in enumerate(members["essay_id"]):
            score_to_band[eid] = start + offset + 1
    df_plot["true_rank_banded"] = df_plot["essay_id"].map(score_to_band)

    color_map = {6: "#2ca02c", 5: "#bcbd22", 4: "#d62728"}

    fig, ax = plt.subplots(figsize=(10, 7))
    xs = [1, 2, 3]
    for _, row in df_plot.iterrows():
        ys = [row["aes_rank"], row["copeland_rank"], row["true_rank_banded"]]
        color = color_map.get(int(row["true_score"]), "#7f7f7f")
        ax.plot(xs, ys, marker="o", color=color, linewidth=1.6, alpha=0.85)
        ax.annotate(row["essay_id"][:7], (xs[0] - 0.05, ys[0]), ha="right", va="center", fontsize=9, color=color)
        ax.annotate(f't={int(row["true_score"])}', (xs[2] + 0.05, ys[2]), ha="left", va="center", fontsize=9, color=color)

    ax.set_xticks(xs)
    ax.set_xticklabels(["AES rank\n(by predicted_score)", "Copeland rank\n(by Claude LLM wins)", "True rank\n(by true_score, banded)"])
    ax.set_ylabel("Rank within bucket (1 = best, 10 = worst)")
    ax.invert_yaxis()
    if with_title:
        ax.set_title("Figure 4 (Claude) -- Rank ladder: AES vs LLM/Copeland vs True (PERSUADE ranks 11-20)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(0.6, 3.4)

    counts_per_score = df_plot["true_score"].value_counts().to_dict()
    handles = [
        plt.Line2D([0], [0], color=color_map[s], marker="o", linewidth=2,
                   label=f"true_score = {s} (n={counts_per_score.get(s, 0)})")
        for s in [6, 5, 4]
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)

    fig.tight_layout()
    name = "fig4_rank_ladder_claude_aes_llm_true"
    out = FIG_DIR / (f"{name}_titled.png" if with_title else f"{name}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    print("wrote", build(with_title=False))
    print("wrote", build(with_title=True))


if __name__ == "__main__":
    main()
