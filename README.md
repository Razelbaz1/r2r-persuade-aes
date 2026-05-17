# r2r-persuade-aes

Code accompanying the workshop paper *Selective Preference Aggregation for Top-k Essay Ranking* (SCaLA-26 submission).

The pipeline implements two-stage essay ranking on the PERSUADE 2.0 corpus: a pointwise AES model produces an initial global ranking, and pairwise LLM comparisons (with social-choice aggregation) refine the local ordering inside a configurable refinement region.

## Pipeline overview

The pipeline runs in six numbered stages. Each stage reads from a per-run directory under `runs/run_<NNN>_seed_<seed>/` and writes its outputs to a stage-specific subdirectory.

| Stage | Script | What it does | Output |
|---|---|---|---|
| PS0 | `scripts/stage_0_download.py` | Download PERSUADE 2.0 train split from Kaggle and write an 80/20 split (vestigial — PS1 reads `data/persuade/train.csv` directly and runs 15-fold CV on the full set). | `stage_0/{essays_train.csv, essays_test.csv}` |
| PS1 | `scripts/stage_1_run_kaggle_notebook.py` | Reproduce a public Kaggle AES notebook locally via papermill. The notebook runs 15-fold StratifiedKFold CV with a LightGBM regressor under a custom QWK objective. Every essay receives an out-of-fold prediction. | `stage_1_aes/predictions.csv` (17,307 rows) |
| PS2 | `scripts/stage_2_select_top_k.py` | Sort by predicted score and select a refinement region (a window of ranks around the top-k cutoff). | `stage_2_top_k/top_k.csv` |
| PS3 | `scripts/stage_3_run_llm_pairwise.py` | Send each unordered essay pair in the refinement region to a prompted LLM judge. Records the win matrix and a per-call audit log. A/B order is randomized per pair to neutralize position bias. | `stage_3_llm/{win_matrix.csv, pairwise_log.csv}` |
| PS3b | `scripts/stage_3b_ensemble_win_matrices.py` | Optional: ensemble two or more win matrices (element-wise sum) into one. Used for multi-judge experiments. | `stage_3_llm/{win_matrix.csv, pairwise_log.csv}` (in the target run dir) |
| PS4 | `scripts/stage_4_fuse_copeland.py` | Aggregate the win matrix into a refined ranking using Copeland scoring (with α tiebreak for ties). | `stage_4_fusion/{copeland_scores.csv, ranking_copeland.csv}` |
| PS5 | `scripts/stage_5_evaluate.py` | Evaluate the AES-only ranking and the Copeland-refined ranking against the human scores. Reports Kendall's τ_b, Spearman's ρ, and pairwise ordering accuracy. | `stage_5_eval/metrics.csv` |

A convenience orchestrator (`scripts/run_all.py`) chains PS0 → PS5 in one command.

## Installation

### Laptop / lightweight setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_persuade.txt
```

This is enough to run PS3-PS5 (the LLM/aggregation/evaluation stages). PS1 (the AES training step) is GPU-heavy and is run on a workstation; PS3+ can run on a laptop using only the API.

### Workstation setup (GPU required for PS1)

```powershell
.\setup_workstation.ps1
```

The script creates a venv with Python 3.12, installs PyTorch with CUDA 12.8 wheels, then the rest of `requirements_persuade.txt`. Tested on Windows 11 + RTX PRO 4000 Blackwell.

### Credentials

PS0 requires Kaggle API credentials at `~/.kaggle/kaggle.json`. PS3 requires `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` environment variables, depending on which provider you select.

## Running the pipeline

End-to-end from scratch (uses gpt-4o-2024-08-06 as the pairwise judge):

```powershell
python scripts/run_all.py --seed 42 --provider openai
```

Individual stages can be invoked directly when iterating on one part of the pipeline without re-running expensive upstream stages:

```powershell
# Just PS3 (assumes PS1+PS2 outputs already exist in the run dir)
python scripts/stage_3_run_llm_pairwise.py `
    --run-dir runs/run_001_seed_42 `
    --provider anthropic `
    --model claude-sonnet-4-6 `
    --temperature 0.0
```

Multi-seed runs over the A/B ordering seed:

```powershell
422..430 | ForEach-Object {
    $seed = $_
    python scripts/stage_3_run_llm_pairwise.py `
        --run-dir "runs/run_seed_42_ab$seed" `
        --provider anthropic --model claude-sonnet-4-6 --temperature 0.0 `
        --seed-ab-order $seed
}
```

## Cache

LLM responses are cached on disk under `cache/llm_responses.sqlite` keyed by `(provider, model_id, temperature, prompt_hash, reasoning_effort)`. Re-running with identical inputs is free and instantaneous. The cache directory is gitignored; preserving it across run-dir cleanups is recommended (LLM responses are the expensive part).

## Layout

```
r2r-persuade-aes/
├── lib/                         # Library modules (judge, voting, metrics, prompts)
│   ├── ab_randomization.py
│   ├── llm_judge.py             # Provider-agnostic judge (OpenAI + Anthropic backends)
│   ├── metrics.py               # tau_b, spearman, pairwise_accuracy
│   ├── prompts.py               # System + user prompt templates
│   └── voting.py                # Copeland scoring
├── scripts/                     # Per-stage entry points + run_all.py orchestrator
│   ├── _common.py
│   ├── run_all.py
│   ├── stage_0_download.py
│   ├── stage_1_run_kaggle_notebook.py
│   ├── stage_1_train_aes_baseline.py
│   ├── stage_2_select_top_k.py
│   ├── stage_3_run_llm_pairwise.py
│   ├── stage_3b_ensemble_win_matrices.py
│   ├── stage_4_fuse_copeland.py
│   └── stage_5_evaluate.py
├── notebooks/                   # Analysis notebooks + figure builders
│   ├── 01_predictions_distribution.ipynb
│   ├── 01_predictions_distribution.py
│   ├── _build_figures_5_and_6.py
│   ├── _build_fig4_claude.py
│   ├── _aggregate_multiseed.py
│   └── figures/                 # PNG figures (300 dpi) referenced by the paper
├── manifest.yaml                # Run metadata (last run config)
├── requirements_persuade.txt    # Python dependencies
├── setup_workstation.ps1        # One-shot venv + CUDA install on Windows workstation
├── .gitignore                   # Excludes runs/, cache/, data/, .venv/, etc.
└── README.md
```

## Reproducing the workshop paper's main result

The headline result reported in the workshop paper:

| Method | Pairwise ordering accuracy on ranks 11-20 |
|---|---:|
| AES baseline (pointwise) | 0.17 |
| Copeland over LLM pairwise (Claude Sonnet 4.6) | 0.48 |

Reproduction:

```powershell
python scripts/run_all.py --seed 42 --provider anthropic --model claude-sonnet-4-6
```

The 17,307 OOF predictions land in `runs/<dir>/stage_1_aes/predictions.csv`; the 10 essays in the refinement region (ranks 11-20) land in `runs/<dir>/stage_2_top_k/top_k.csv`; the 45 LLM pairwise judgments and the resulting win matrix land in `runs/<dir>/stage_3_llm/`; the Copeland-refined ranking lands in `runs/<dir>/stage_4_fusion/ranking_copeland.csv`; the headline metric lands in `runs/<dir>/stage_5_eval/metrics.csv`.

## Datasets and external resources

- **PERSUADE 2.0 train.csv** — Kaggle competition *Learning Agency Lab Automated Essay Scoring 2.0*. Download via `kaggle competitions download -c learning-agency-lab-automated-essay-scoring-2 -f train.csv` (requires accepting the competition rules).
- **datafan07/some-extra-features-cv-0-83** — public Kaggle notebook used as the AES baseline. Pulled at PS1 setup; not redistributed in this repo.

## License

TBD.
