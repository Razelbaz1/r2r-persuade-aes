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
5. **Inject a final cell** that exports `train[['essay_id', 'oof']]`
   to a CSV. The notebook's 15-fold StratifiedKFold loop fills the
   `oof` column with out-of-fold regression predictions; every essay
   is predicted by a model that did not see it during training.
6. **Register a Jupyter kernel** named `persuade_aes` pointing at the
   venv's Python. Idempotent -- running multiple times is safe.
7. **Execute the patched notebook** via papermill, writing the
   executed copy to `<run-dir>/stage_1_aes/notebook_executed.ipynb`.
8. **Post-process**: merge the OOF CSV with `train.csv` (for
   `essay_text` and `score`) and emit
   `<run-dir>/stage_1_aes/predictions.csv` in the pipeline's schema
   (`essay_id, essay_text, predicted_score, true_score`).

Expected runtime on LAB-LIHID (RTX PRO 4000 Blackwell): ~45-90 minutes,
dominated by the 5 DeBERTa-small inference passes in the notebook's
feedback-feature stage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat
import pandas as pd
import papermill as pm

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


# === Step 3-5: patch notebook ===

def patch_notebook(
    nb: nbformat.NotebookNode,
    oof_output_path: Path,
) -> nbformat.NotebookNode:
    """Rewrite Kaggle paths, neutralize pip-install cell, fix pandas 2.x dtype
    compat, inject OOF export."""
    mappings = _path_mappings()
    oof_path_posix = str(oof_output_path).replace("\\", "/")
    patched = 0
    neutralized = 0
    dtype_patched = 0

    for cell in nb.cells:
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

        cell.source = src

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
        f"applied {dtype_patched} pandas dtype fix(es)."
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


# === Steps 7-8: execute + post-process ===

def main(run_dir: Path) -> Path:
    out_dir = stage_dir(run_dir, "stage_1_aes")

    print("=== Step 1: locate / pull notebook ===")
    notebook_path = ensure_notebook_pulled()
    print(f"Notebook: {notebook_path}")

    print("=== Step 2: dummy test artifacts ===")
    ensure_dummy_test_files()

    print("=== Step 3-5: patch notebook ===")
    nb = nbformat.read(notebook_path, as_version=4)
    oof_path = out_dir / "oof_predictions.csv"
    nb = patch_notebook(nb, oof_path)

    patched_in = out_dir / "notebook_patched.ipynb"
    nbformat.write(nb, patched_in)
    print(f"Patched notebook saved to {patched_in}")

    print("=== Step 6: register kernel ===")
    ensure_kernel_registered()

    print("=== Step 7: execute notebook via papermill (this is the slow part) ===")
    executed_out = out_dir / "notebook_executed.ipynb"
    pm.execute_notebook(
        str(patched_in),
        str(executed_out),
        kernel_name=KERNEL_NAME,
        cwd=str(REPO_ROOT),
        progress_bar=True,
    )
    print(f"Executed notebook saved to {executed_out}")

    print("=== Step 8: post-process to predictions.csv ===")
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
    return predictions_path


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="PS1 -- run datafan07's Kaggle notebook locally and emit predictions.csv"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = resolve_or_create_run_dir(seed=-1, explicit=args.run_dir)
    out_path = main(run_dir=run_dir)
    print(f"PS1 done. {out_path}")


if __name__ == "__main__":
    cli()
