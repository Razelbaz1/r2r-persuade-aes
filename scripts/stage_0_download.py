"""PS0 — Download the PERSUADE 2 training data and build the 80/20 split.

We pull the **training data** (only) from the Kaggle competition
`learning-agency-lab-automated-essay-scoring-2`. The competition's
test set carries no public labels — per Lihi's playbook (step 1)
and step 3 of `for_raz.tex` we therefore work with the labeled
training set only, and produce our own 80/20 train/test split with
`random_state=42`.

The competition data ships as `train.csv` with three columns:
`essay_id`, `full_text`, `score`. We rename `full_text` → `essay_text`
and `score` → `true_score` to match the schema the rest of the
pipeline expects (`essay_id` is already correct).

Why the competition data, not `nbroad/persaude-corpus-2`: the Kaggle
notebooks we plan to reproduce in PS1 expect the competition layout
(`/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv`).
Using the competition source minimizes path-patching when those
notebooks run locally. See `memory/persuade_dataset_choice_2026_05_13.md`.

The corpus is downloaded once to `data/persuade/` at the repo root and
reused across run-dirs; only the per-seed split is written into
`<run-dir>/stage_0/`. The Kaggle CLI must be configured (see
`README.md`) AND you must have accepted the competition rules on
Kaggle before this script can download the data.
"""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from _common import resolve_or_create_run_dir, stage_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "persuade"
KAGGLE_COMPETITION = "learning-agency-lab-automated-essay-scoring-2"
COMPETITION_TRAIN_FILE = "train.csv"


def _train_csv_path() -> Path:
    return DATA_DIR / COMPETITION_TRAIN_FILE


def _ensure_dataset_downloaded() -> Path:
    """Pull `train.csv` from the competition if it is not on disk.

    Returns the path to the unzipped train.csv. Idempotent — if the
    file is already present, the download is skipped.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_csv = _train_csv_path()
    if train_csv.exists():
        return train_csv

    print(
        f"Downloading {COMPETITION_TRAIN_FILE} from competition "
        f"{KAGGLE_COMPETITION} into {DATA_DIR} ..."
    )
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            KAGGLE_COMPETITION,
            "-f",
            COMPETITION_TRAIN_FILE,
            "-p",
            str(DATA_DIR),
        ],
        check=True,
    )

    # The competition CLI delivers a .zip even for single-file pulls.
    zip_path = DATA_DIR / f"{COMPETITION_TRAIN_FILE}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()

    if not train_csv.exists():
        raise FileNotFoundError(
            f"{COMPETITION_TRAIN_FILE} not found after download under {DATA_DIR}. "
            f"Files present: {[p.name for p in DATA_DIR.iterdir()]}. "
            "Have you accepted the competition rules on Kaggle?"
        )
    return train_csv


def _normalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename the corpus columns into the pipeline's schema.

    Required output columns: `essay_id`, `essay_text`, `true_score`.
    Falls back to common alternative names if PERSUADE's columns change.
    """
    # Competition schema is `essay_id`, `full_text`, `score` — listed
    # first. Older nbroad-dataset names (`essay_id_comp`,
    # `holistic_essay_score`) kept as fallbacks for forward compatibility.
    id_candidates = ("essay_id", "essay_id_comp", "id")
    text_candidates = ("full_text", "essay_text", "text", "essay")
    score_candidates = (
        "score",
        "holistic_essay_score",
        "holistic_score",
        "true_score",
        "label",
    )

    def find(cands: tuple[str, ...]) -> str:
        for c in cands:
            if c in raw.columns:
                return c
        raise KeyError(f"None of {cands} found in columns: {list(raw.columns)}")

    id_col = find(id_candidates)
    text_col = find(text_candidates)
    score_col = find(score_candidates)
    out = raw[[id_col, text_col, score_col]].rename(
        columns={id_col: "essay_id", text_col: "essay_text", score_col: "true_score"}
    )
    out["true_score"] = out["true_score"].astype(float)
    return out.reset_index(drop=True)


def main(run_dir: Path, seed: int, test_size: float = 0.20) -> tuple[Path, Path]:
    csv_path = _ensure_dataset_downloaded()
    print(f"Loading corpus from {csv_path}")
    raw = pd.read_csv(csv_path)
    df = _normalize_columns(raw)
    print(f"Loaded {len(df):,} essays. Columns: {list(df.columns)}")

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True
    )

    out_dir = stage_dir(run_dir, "stage_0")
    train_path = out_dir / "essays_train.csv"
    test_path = out_dir / "essays_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Split: train={len(train_df):,}, test={len(test_df):,}")
    return train_path, test_path


def cli() -> None:
    parser = argparse.ArgumentParser(description="PS0 — download + split")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_dir = resolve_or_create_run_dir(seed=args.seed, explicit=args.run_dir)
    train_path, test_path = main(run_dir=run_dir, seed=args.seed)
    print(f"PS0 done. Wrote {train_path} and {test_path}")


if __name__ == "__main__":
    cli()
