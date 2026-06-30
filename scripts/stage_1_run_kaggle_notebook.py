"""PS1 -- Reproduce datafan07's Kaggle notebook locally and emit predictions.

This is the canonical PS1 for the PERSUADE/LLM Phase-1 pipeline. The
older `stage_1_train_aes_baseline.py` (our hand-written TF-IDF +
LightGBM) is kept in the tree for reference but no longer runs in
`run_all.py`.

What this wrapper does, step by step:

1. **Pull the notebook** `datafan07/some-extra-features-cv-0-83`
   from Kaggle into `experiments/persuade_aes/notebooks_pulled/` if it
   is not already on disk. Kaggle credentials must be configured.
2. **Provide dummy test artifacts**: the notebook reads
   `test.csv` and `sample_submission.csv` from the competition input
   directory. The competition's test labels are not public; for our
   purposes (OOF predictions on the labeled train set) we generate
   tiny placeholder files alongside the real `train.csv` so the
   notebook can run untouched.
3. **Patch Kaggle paths to local paths**: rewrite every
   `/kaggle/input/...` reference in the notebook source to point at
   `data/persuade/` and `data/persuade_aux/` on the local disk.
4. **Neutralize the offline pip-install cell**: the notebook's first
   code cell installs HuggingFace wheels from a Kaggle dataset. Those
   wheels are for Linux + cp310; on the workstation we have the same
   packages installed in the venv via PyPI. The cell is replaced with
   a no-op.
5. **Substitute the seed (and fold count)**: replace `random_state=42`
   (used by both `StratifiedKFold` and `LGBMRegressor`, both inside the
   same CV cell) with the seed supplied via `--seed`. Without this, every
   run uses the notebook's hard-coded 42. Likewise, when `--n-folds`
   differs from 15, rewrite `n_splits=15` in the CV cell so the OOF
   predictions come from the requested number of folds.
6. **Cache Stage A features** (deterministic feature-extraction
   outputs: `merged_df`, `merged_df_test`, plus the four small auxiliary
   DataFrames). When `--features-cache PATH` is supplied:
     - If the cache file exists, the wrapper neutralizes the data-load
       cell and all Stage A cells, and injects a load cell right after
       the imports cell. The CV cell then runs on cached features.
     - If the cache file does not exist, the wrapper injects a save
       cell right before the CV cell. The save persists Stage A
       outputs after they are computed once, so subsequent seeds skip
       the expensive feature pass.
   Use `--no-cache` to disable the cache logic entirely.
7. **Inject a final cell** that exports `train[['essay_id', 'oof']]`
   to a CSV. The notebook's N-fold StratifiedKFold loop (N = `--n-folds`,
   default 15) fills the `oof` column with out-of-fold regression
   predictions; every essay is predicted by a model that did not see it
   during training.
8. **Register a Jupyter kernel** named `persuade_aes` pointing at the
   venv's Python. Idempotent -- running multiple times is safe.
9. **Execute the patched notebook** via papermill, writing the
   executed copy to `<run-dir>/stage_1_aes/notebook_executed.ipynb`.
10. **Post-process**: merge the OOF CSV with `train.csv` (for
    `essay_text` and `score`) and emit
    `<run-dir>/stage_1_aes/predictions.csv` in the pipeline's schema
    (`essay_id, essay_text, predicted_score, true_score`). Then write
    `<run-dir>/stage_1_aes/predictions_with_fold.csv` (same rows plus a
    `fold` column reproduced from `--seed` and `--n-folds`), a long-format
    `per_fold_metrics.csv` (one row per fold: QWK, Spearman rho, Kendall
    tau-b, pairwise accuracy), and a `per_fold_tables/` directory with one
    predicted-score-ranked CSV per fold. For canonical 15-fold runs it
    also appends a row to `experiments/persuade_aes/seed_sweep_log.csv`.

Expected runtime on LAB-LIHID (RTX PRO 4000 Blackwell):
- Cache miss (full Stage A + CV): ~28 minutes
- Cache hit (only CV): ~5-6 minutes
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import papermill as pm
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

from _common import (
    PACKAGE_ROOT,
    resolve_or_create_run_dir,
    stage_dir,
)

# === Constants ===

REPO_ROOT = PACKAGE_ROOT.parents[1]
PERSUADE_DATA = REPO_ROOT / "data" / "persuade"
PERSUADE_AUX = REPO_ROOT / "data" / "persuade_aux"
NOTEBOOKS_DIR = PACKAGE_ROOT / "notebooks_pulled"

KAGGLE_NOTEBOOK = "datafan07/some-extra-features-cv-0-83"
NOTEBOOK_FILE_GLOB = "some-extra-features-cv-0-83*.ipynb"
KERNEL_NAME = "persuade_aes"

# Paths to substitute inside the notebook. Order matters: longest prefix
# first so we do not accidentally rewrite a shorter prefix inside a
# longer path.
def _path_mappings() -> list[tuple[str, str]]:
    def posix(p: Path) -> str:
        return str(p).replace("\\", "/")

    return [
        # Aux datasets first (longer prefixes)
        ("/kaggle/input/sent-debsmall", posix(PERSUADE_AUX / "sent_debsmall")),
        ("/kaggle/input/feedback-data", posix(PERSUADE_AUX / "feedback_data")),
        # Competition data
        (
            "/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
            posix(PERSUADE_DATA),
        ),
    ]


# === Step 1: pull notebook ===

def ensure_notebook_pulled() -> Path:
    """Locate the notebook on disk; pull from Kaggle if missing."""
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    matches = list(NOTEBOOKS_DIR.glob(NOTEBOOK_FILE_GLOB))
    if matches:
        return matches[0]

    print(f"Pulling {KAGGLE_NOTEBOOK} into {NOTEBOOKS_DIR} ...")
    subprocess.run(
        ["kaggle", "kernels", "pull", KAGGLE_NOTEBOOK, "-p", str(NOTEBOOKS_DIR)],
        check=True,
    )
    matches = list(NOTEBOOKS_DIR.glob(NOTEBOOK_FILE_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"Notebook not found after `kaggle kernels pull`. "
            f"Files present in {NOTEBOOKS_DIR}: {list(NOTEBOOKS_DIR.iterdir())}"
        )
    return matches[0]


# === Step 2: dummy test artifacts ===

def ensure_dummy_test_files() -> None:
    """Create test.csv + sample_submission.csv next to train.csv if absent.

    The Kaggle competition's test labels are private. For our pipeline
    we use OOF predictions on train.csv exclusively -- the test set
    that the notebook computes is throwaway. Five rows is enough to
    keep the notebook happy.
    """
    train_csv = PERSUADE_DATA / "train.csv"
    test_csv = PERSUADE_DATA / "test.csv"
    sample_sub_csv = PERSUADE_DATA / "sample_submission.csv"

    if test_csv.exists() and sample_sub_csv.exists():
        return

    if not train_csv.exists():
        raise FileNotFoundError(
            f"train.csv missing at {train_csv}. Run PS0 (stage_0_download) first."
        )

    train = pd.read_csv(train_csv)
    head = train.head(5).copy()

    # Real competition test has only essay_id + full_text (no score).
    head[["essay_id", "full_text"]].to_csv(test_csv, index=False)

    pd.DataFrame({"essay_id": head["essay_id"], "score": 3}).to_csv(
        sample_sub_csv, index=False
    )
    print(f"Created dummy {test_csv.name} and {sample_sub_csv.name} (5 rows each).")


# === Step 3-6: patch notebook ===
#
# Cache LOAD mode uses POSITIONAL detection (not content keywords).
# Every code cell strictly between the imports cell (the first cell
# containing `import polars`) and the CV cell (the first cell
# containing `skf = StratifiedKFold`) is neutralized. The imports
# cell, the wrapper-injected cache-load cell, the CV cell, and every
# cell after the CV cell are preserved.
#
# Earlier versions of this wrapper used a content-keyword list to
# identify Stage A cells. That approach missed cells that consumed
# Stage A variables without containing any of the producer markers
# (e.g. a ratio-features cell referencing `train_linguistic`), and
# also misclassified the imports cell when its `from X import Y`
# lines happened to contain the same substrings. Positional detection
# avoids both failure modes.


def patch_notebook(
    nb: nbformat.NotebookNode,
    oof_output_path: Path,
    seed: int = 42,
    features_cache_path: Path | None = None,
    n_folds: int = 15,
) -> nbformat.NotebookNode:
    """Rewrite Kaggle paths, neutralize pip-install cell, fix pandas 2.x dtype
    compat, optionally substitute the CV seed, optionally wire in a
    features cache, and inject the OOF export.

    Parameters
    ----------
    nb : the parsed notebook (mutated in place).
    oof_output_path : where the injected final cell writes the OOF CSV.
    seed : random_state used for both StratifiedKFold and LGBMRegressor
        in the CV cell. When 42 (the notebook's hard-coded default) no
        substitution is performed.
    features_cache_path : optional path for the Stage A features pickle.
        If the file exists, Stage A cells + data-load cell are
        neutralized and a load cell is injected after the imports cell.
        If the file does not exist, a save cell is injected before the
        CV cell. If None, no cache logic is applied.
    n_folds : number of StratifiedKFold splits. When different from 15
        (the notebook's hard-coded default) the wrapper rewrites
        `n_splits=15` in the CV cell.
    """
    mappings = _path_mappings()
    oof_path_posix = str(oof_output_path).replace("\\", "/")
    patched = 0
    neutralized = 0
    dtype_patched = 0
    seed_patched = 0
    folds_patched = 0

    # Cache-mode flags.
    cache_load_mode = (
        features_cache_path is not None and features_cache_path.exists()
    )
    cache_save_mode = (
        features_cache_path is not None and not features_cache_path.exists()
    )
    cache_path_posix = (
        str(features_cache_path).replace("\\", "/")
        if features_cache_path is not None
        else None
    )

    # Cell-index bookkeeping for the cache logic. The actual list of
    # cells to neutralize is built positionally AFTER the first pass
    # (see cache LOAD block below) because it needs both anchors to
    # be known.
    imports_cell_idx: int | None = None
    cv_cell_idx: int | None = None

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = cell.source

        # Neutralize the offline pip-install cell -- detected by the
        # `!pip install --no-index` prefix that this cell uses.
        if "!pip install --no-index" in src:
            cell.source = (
                "# Wrapper: offline pip install neutralized -- "
                "packages are already installed in the venv.\n"
                "pass\n"
            )
            neutralized += 1
            continue

        # Substitute every known Kaggle path.
        for kaggle_path, local_path in mappings:
            if kaggle_path in src:
                src = src.replace(kaggle_path, local_path)
                patched += 1

        # Pandas 2.x compatibility: the notebook initializes train['oof']
        # with the int literal 0, which creates an int64 column. Pandas
        # 2.x then refuses to coerce when float predictions are written
        # via .loc and raises TypeError "Invalid value '[...]' for dtype
        # 'int64'". Older pandas silently upcast. We patch the literal
        # to 0.0 so the column starts as float64.
        if "train['oof'] = 0\n" in src:
            src = src.replace(
                "train['oof'] = 0\n",
                "train['oof'] = 0.0  # patched by wrapper for pandas 2.x dtype safety\n",
            )
            dtype_patched += 1

        # Seed substitution: `random_state=42` appears twice -- once in
        # StratifiedKFold and once in LGBMRegressor -- both inside the
        # same CV cell. A blanket replace is safe.
        if seed != 42 and "random_state=42" in src:
            src = src.replace("random_state=42", f"random_state={seed}")
            seed_patched += src.count(f"random_state={seed}")

        # Fold-count substitution: the notebook hard-codes
        # `StratifiedKFold(n_splits=15, ...)` (single occurrence). Repoint
        # it to --n-folds so the OOF predictions come from the requested
        # number of folds. Must stay in sync with the fold reproduction in
        # `_write_predictions_with_fold` and `_write_per_fold_metrics`.
        if n_folds != 15 and "n_splits=15" in src:
            src = src.replace("n_splits=15", f"n_splits={n_folds}")
            folds_patched += src.count(f"n_splits={n_folds}")

        cell.source = src

        # After source mutations, track the two cache-mode anchors.
        if "import polars" in src and imports_cell_idx is None:
            imports_cell_idx = idx
        if "skf = StratifiedKFold" in src and cv_cell_idx is None:
            cv_cell_idx = idx

    # Apply cache logic AFTER the first pass so cell indices are stable.
    cache_msg = "no cache logic applied"
    if cache_load_mode:
        if imports_cell_idx is None:
            raise RuntimeError(
                "Could not locate imports cell (missing `import polars` marker). "
                "Cache load requires the imports cell to anchor the load cell."
            )
        if cv_cell_idx is None:
            raise RuntimeError(
                "Could not locate CV cell (missing `skf = StratifiedKFold` "
                "marker). Cache load requires the CV cell to bound the "
                "positional neutralization window."
            )
        if cv_cell_idx <= imports_cell_idx:
            raise RuntimeError(
                f"Notebook structure unexpected: CV cell (idx {cv_cell_idx}) "
                f"is not strictly after imports cell (idx {imports_cell_idx}). "
                f"Cannot determine the Stage A range."
            )
        # Positional neutralization: every code cell strictly between
        # imports and CV. Markdown cells are left alone. The pip-install
        # cell, if present in this range, was already converted to
        # `pass` earlier in the loop -- re-writing it to `pass` is a
        # no-op.
        cells_to_neutralize = [
            i
            for i in range(imports_cell_idx + 1, cv_cell_idx)
            if nb.cells[i].cell_type == "code"
        ]
        for idx in sorted(set(cells_to_neutralize), reverse=True):
            nb.cells[idx].source = (
                "# Wrapper: code cell between imports and CV neutralized "
                "(cache load mode).\npass\n"
            )
        load_src = (
            "# Wrapper-injected: load cached Stage A features.\n"
            "import pickle as _pickle\n"
            f"_cache_path = r'{cache_path_posix}'\n"
            "try:\n"
            "    with open(_cache_path, 'rb') as _f:\n"
            "        _cache = _pickle.load(_f)\n"
            "except (FileNotFoundError, EOFError, _pickle.UnpicklingError) as _exc:\n"
            "    raise RuntimeError(\n"
            "        f'Features cache at {_cache_path} is missing or corrupted ({_exc!r}). '\n"
            "        f'Delete it and rerun without an existing cache to recompute.'\n"
            "    )\n"
            "train = _cache['train']\n"
            "test = _cache['test']\n"
            "sample_submission = _cache['sample_submission']\n"
            "merged_df = _cache['merged_df']\n"
            "merged_df_test = _cache['merged_df_test']\n"
            "feedback_predictions_df = _cache['feedback_predictions_df']\n"
            "test_feedback_predictions_df = _cache['test_feedback_predictions_df']\n"
            "print(f'[wrapper] Loaded cached features: merged_df={merged_df.shape}')\n"
        )
        nb.cells.insert(
            imports_cell_idx + 1, nbformat.v4.new_code_cell(source=load_src)
        )
        cache_msg = (
            f"cache LOAD mode: neutralized {len(set(cells_to_neutralize))} "
            f"code cells in range ({imports_cell_idx}, {cv_cell_idx}) "
            f"and injected load cell after index {imports_cell_idx}"
        )
    elif cache_save_mode:
        if cv_cell_idx is None:
            raise RuntimeError(
                "Could not locate the CV cell (missing `skf = StratifiedKFold` "
                "marker). Cache save requires the CV cell to anchor the save cell."
            )
        save_src = (
            "# Wrapper-injected: save Stage A features for seed-sweep reuse.\n"
            "import pickle as _pickle\n"
            f"_cache_path = r'{cache_path_posix}'\n"
            "with open(_cache_path, 'wb') as _f:\n"
            "    _pickle.dump({\n"
            "        'train': train,\n"
            "        'test': test,\n"
            "        'sample_submission': sample_submission,\n"
            "        'merged_df': merged_df,\n"
            "        'merged_df_test': merged_df_test,\n"
            "        'feedback_predictions_df': feedback_predictions_df,\n"
            "        'test_feedback_predictions_df': test_feedback_predictions_df,\n"
            "    }, _f)\n"
            "print(f'[wrapper] Saved features cache to {_cache_path} '\n"
            "      f'(merged_df={merged_df.shape})')\n"
        )
        nb.cells.insert(cv_cell_idx, nbformat.v4.new_code_cell(source=save_src))
        cache_msg = (
            f"cache SAVE mode: injected save cell before CV cell (index "
            f"{cv_cell_idx})"
        )

    # Inject a final cell to export the OOF predictions.
    extract_src = (
        "# Wrapper-injected: export OOF predictions for the PERSUADE pipeline.\n"
        "import pandas as _wrapper_pd\n"
        "_wrapper_oof = train[['essay_id', 'oof']].copy()\n"
        f"_wrapper_oof.to_csv(r'{oof_path_posix}', index=False)\n"
        f"print(f'Wrote {{len(_wrapper_oof):,}} OOF predictions to {oof_path_posix}')\n"
    )
    nb.cells.append(nbformat.v4.new_code_cell(source=extract_src))

    print(
        f"Patched paths in {patched} cells; "
        f"neutralized {neutralized} pip-install cell(s); "
        f"applied {dtype_patched} pandas dtype fix(es); "
        f"applied {seed_patched} seed substitution(s); "
        f"applied {folds_patched} fold-count substitution(s); "
        f"{cache_msg}."
    )
    return nb


# === Step 6: register Jupyter kernel ===

def ensure_kernel_registered() -> None:
    """Idempotently register the active venv's Python as a Jupyter kernel.

    Papermill needs a `kernel_name` that maps to a discoverable
    `kernelspec`. We install one named `persuade_aes` pointing at the
    current `sys.executable` -- safe to re-run.
    """
    print(f"Registering Jupyter kernel '{KERNEL_NAME}' (interpreter: {sys.executable})")
    subprocess.run(
        [
            sys.executable, "-m", "ipykernel", "install",
            "--user",
            "--name", KERNEL_NAME,
            "--display-name", "PERSUADE-AES venv",
        ],
        check=True,
    )


# === Steps 7-10: execute + post-process ===

def main(
    run_dir: Path,
    seed: int = 42,
    features_cache_path: Path | None = None,
    n_folds: int = 15,
) -> Path:
    out_dir = stage_dir(run_dir, "stage_1_aes")

    print("=== Step 1: locate / pull notebook ===")
    notebook_path = ensure_notebook_pulled()
    print(f"Notebook: {notebook_path}")

    print("=== Step 2: dummy test artifacts ===")
    ensure_dummy_test_files()

    print(
        f"=== Steps 3-6: patch notebook "
        f"(seed={seed}, n_folds={n_folds}, "
        f"features_cache={features_cache_path or 'disabled'}) ==="
    )
    nb = nbformat.read(notebook_path, as_version=4)
    oof_path = out_dir / "oof_predictions.csv"
    nb = patch_notebook(
        nb,
        oof_path,
        seed=seed,
        features_cache_path=features_cache_path,
        n_folds=n_folds,
    )

    patched_in = out_dir / "notebook_patched.ipynb"
    nbformat.write(nb, patched_in)
    print(f"Patched notebook saved to {patched_in}")

    print("=== Step 7: register kernel ===")
    ensure_kernel_registered()

    print("=== Step 8: execute notebook via papermill (this is the slow part) ===")
    executed_out = out_dir / "notebook_executed.ipynb"
    pm.execute_notebook(
        str(patched_in),
        str(executed_out),
        kernel_name=KERNEL_NAME,
        cwd=str(REPO_ROOT),
        progress_bar=True,
    )
    print(f"Executed notebook saved to {executed_out}")

    print("=== Step 9: post-process to predictions.csv ===")
    train = pd.read_csv(PERSUADE_DATA / "train.csv")
    oof = pd.read_csv(oof_path)
    merged = train.merge(oof, on="essay_id")

    predictions = pd.DataFrame({
        "essay_id": merged["essay_id"],
        "essay_text": merged["full_text"],
        "predicted_score": merged["oof"],
        "true_score": merged["score"],
    })

    predictions_path = out_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    print(f"Wrote {len(predictions):,} rows to {predictions_path}")

    print("=== Step 10: write predictions_with_fold.csv + per-fold metrics + per-fold ranked tables ===")
    fold_csv = _write_predictions_with_fold(predictions_path, seed=seed, n_folds=n_folds)
    _write_per_fold_metrics(fold_csv, n_folds=n_folds)
    _write_per_fold_ranked_tables(fold_csv, n_folds=n_folds)

    # The wide seed-sweep log has a fixed 15-fold schema; only append to it
    # for canonical 15-fold runs. n-fold runs are summarized by
    # per_fold_metrics.csv instead.
    if n_folds == 15:
        _append_seed_sweep_log(
            log_path=PACKAGE_ROOT / "seed_sweep_log.csv",
            predictions_path=predictions_path,
            seed=seed,
            run_dir=run_dir,
        )

    return predictions_path


# === Post-processing helpers (per-seed fold reproduction + sweep log) ===

def _write_predictions_with_fold(
    predictions_path: Path, seed: int, n_folds: int = 15
) -> Path:
    """Reproduce the StratifiedKFold(n_splits=n_folds, shuffle=True,
    random_state=seed) split on the predictions row order, add a `fold`
    column, and write `predictions_with_fold.csv` next to predictions.csv.

    The notebook itself does not persist the fold assignment. We reproduce
    it from `(seed, n_folds)` -- the same StratifiedKFold parameters the
    patched CV cell uses -- so the reproduced folds match the folds that
    produced the OOF predictions. (Row order is preserved because
    predictions.csv is a left-merge of train.csv with the OOF CSV.)
    """
    df = pd.read_csv(predictions_path).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_id = np.full(len(df), -1, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(
        skf.split(np.zeros(len(df)), df["true_score"].values)
    ):
        fold_id[val_idx] = fold_idx
    if not (fold_id >= 0).all():
        raise RuntimeError("Fold reproduction failed: some essays unassigned.")
    df["fold"] = fold_id
    df = df[["essay_id", "fold", "predicted_score", "true_score", "essay_text"]]
    out = predictions_path.with_name("predictions_with_fold.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")
    return out


def _pairwise_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of essay pairs whose predicted order matches the true order,
    over pairs with *distinct* true scores. Tied-true pairs carry no
    ground-truth order and are excluded. This is the whole-fold analogue of
    the paper's within-window pairwise ordering accuracy.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    iu = np.triu_indices(len(yt), k=1)
    dt = np.sign(yt[iu[0]] - yt[iu[1]])
    dp = np.sign(yp[iu[0]] - yp[iu[1]])
    mask = dt != 0
    n_pairs = int(mask.sum())
    if n_pairs == 0:
        return float("nan")
    return int((dp[mask] == dt[mask]).sum()) / n_pairs


def _write_per_fold_metrics(fold_csv: Path, n_folds: int) -> Path:
    """Write a long-format per-fold accuracy table -- one row per fold with
    QWK, Spearman rho, Kendall tau-b, and pairwise accuracy of the scoring
    model's predicted scores against the true scores within that fold.

    The four measure related but distinct things: QWK on the rounded scores
    (the model's training objective), Spearman and Kendall tau-b on the
    ranking (tau-b handles the heavy ties in the 1--6 true scores), and
    pairwise accuracy on the fraction of correctly ordered non-tied pairs.
    """
    df = pd.read_csv(fold_csv)
    rows = []
    for fold in range(n_folds):
        sub = df[df["fold"] == fold]
        yt = sub["true_score"].to_numpy()
        yp = sub["predicted_score"].to_numpy()
        qwk = cohen_kappa_score(
            yt, np.clip(np.round(yp).astype(int), 1, 6), weights="quadratic"
        )
        rho, _ = spearmanr(yp, yt)
        tau_b, _ = kendalltau(yp, yt, variant="b")
        rows.append(
            {
                "fold": fold,
                "n_essays": len(sub),
                "qwk": round(float(qwk), 6),
                "spearman_rho": round(float(rho), 6),
                "kendall_tau_b": round(float(tau_b), 6),
                "pairwise_accuracy": round(_pairwise_accuracy(yp, yt), 6),
            }
        )
    out = fold_csv.with_name("per_fold_metrics.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote per-fold metrics for {n_folds} folds to {out}")
    return out


def _write_per_fold_ranked_tables(fold_csv: Path, n_folds: int) -> Path:
    """Emit one ranked CSV per fold (sorted by predicted_score, descending)
    into a `per_fold_tables/` directory. Each file is a self-contained
    experiment table for the r/k refinement sweeps. Columns: rank_position,
    essay_id, fold, predicted_score, true_score, essay_text.
    """
    df = pd.read_csv(fold_csv)
    out_dir = fold_csv.with_name("per_fold_tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["essay_id", "fold", "predicted_score", "true_score", "essay_text"]
    for fold in range(n_folds):
        sub = (
            df[df["fold"] == fold]
            .sort_values("predicted_score", ascending=False)
            .reset_index(drop=True)
        )
        sub.insert(0, "rank_position", np.arange(1, len(sub) + 1))
        sub[["rank_position"] + cols].to_csv(
            out_dir / f"fold_{fold:02d}_ranked.csv", index=False
        )
    print(f"Wrote {n_folds} per-fold ranked tables to {out_dir}")
    return out_dir


def _append_seed_sweep_log(
    log_path: Path,
    predictions_path: Path,
    seed: int,
    run_dir: Path,
) -> None:
    """Compute 15 per-fold QWKs + the mean QWK for this seed and append
    one row to the global sweep log. Creates the file with a header on
    the first run.

    QWK is computed on rounded-and-clipped predictions per fold, then
    averaged across the 15 folds.
    """
    df = pd.read_csv(predictions_path).reset_index(drop=True)
    y_true = df["true_score"].values
    y_pred = df["predicted_score"].values
    skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=seed)
    fold_qwks: list[float] = []
    for _, val_idx in skf.split(np.zeros(len(df)), y_true):
        yt = y_true[val_idx]
        yp = np.clip(np.round(y_pred[val_idx]).astype(int), 1, 6)
        fold_qwks.append(cohen_kappa_score(yt, yp, weights="quadratic"))
    mean_qwk = float(np.mean(fold_qwks))

    header = (
        ["seed"]
        + [f"fold_{i}_qwk" for i in range(15)]
        + ["mean_qwk", "n_total", "run_dir", "finished_at"]
    )
    row = (
        [seed]
        + [f"{q:.6f}" for q in fold_qwks]
        + [
            f"{mean_qwk:.6f}",
            len(df),
            str(run_dir),
            datetime.utcnow().isoformat(timespec="seconds"),
        ]
    )
    new_file = not log_path.exists()
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)
    print(f"Appended seed {seed} row to {log_path} (mean QWK = {mean_qwk:.4f})")


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="PS1 -- run datafan07's Kaggle notebook locally and emit predictions.csv"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random_state for StratifiedKFold + LGBMRegressor in the CV cell "
             "(default: 42, the notebook's original value).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=15,
        help="Number of StratifiedKFold splits in the CV cell (default: 15, "
             "the notebook's original value). Use 30 for the finer-grained "
             "per-fold experiment.",
    )
    parser.add_argument(
        "--features-cache",
        type=Path,
        default=PACKAGE_ROOT / "_features_cache.pkl",
        help="Path to the Stage A features cache. If the file exists, the "
             "notebook is patched to load it and skip feature extraction. "
             "If it does not exist, the notebook is patched to save it "
             "after Stage A. Default: experiments/persuade_aes/_features_cache.pkl.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the features cache logic (always recompute Stage A).",
    )
    args = parser.parse_args()
    run_dir = resolve_or_create_run_dir(seed=args.seed, explicit=args.run_dir)
    features_cache = None if args.no_cache else args.features_cache
    out_path = main(
        run_dir=run_dir,
        seed=args.seed,
        features_cache_path=features_cache,
        n_folds=args.n_folds,
    )
    print(f"PS1 done (seed={args.seed}). {out_path}")


if __name__ == "__main__":
    cli()
