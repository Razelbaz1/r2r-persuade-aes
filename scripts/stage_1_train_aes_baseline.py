"""[DEPRECATED 2026-05-14] PS1 -- AES baseline (TF-IDF + LightGBM).

Superseded by `stage_1_run_kaggle_notebook.py`, which reproduces
`datafan07/some-extra-features-cv-0-83` (CV 0.83) locally instead of
training a baseline from scratch. The new path is in line with Lihi's
2026-05-13 directive: reproduce a published competition notebook
rather than build our own.

This file is kept in the tree for reference and for the hypothetical
case where we want to "reinvent the wheel" with a simpler hand-written
baseline. `run_all.py` does NOT call this module any more.

---

Original docstring follows:

Adapts the public Kaggle notebook Lihi linked in step 2 of the playbook
(`kaggle.com/code/ye11725/tfidf-lgbm-baseline-cv-0-799-lb-0-799`).
The implementation is intentionally compact -- the goal is a workable
baseline on top of which the LLM refinement is layered, not a tuned
AES system. Tuning is a Phase 3 concern.

Inputs (read from `<run-dir>/stage_0/`):
- `essays_train.csv`, `essays_test.csv` (columns: essay_id, essay_text, true_score)

Outputs (written to `<run-dir>/stage_1_aes/`):
- `model.pkl` — joblib-pickled (TfidfVectorizer, LGBMRegressor) tuple
- `predictions.csv` — (essay_id, essay_text, predicted_score, true_score)
- `train_metadata.json` — feature dimensionality + training time
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from _common import resolve_or_create_run_dir, stage_dir


def _train(
    train_texts: list[str],
    train_scores: list[float],
    seed: int,
) -> tuple[TfidfVectorizer, "object", dict]:
    """Fit TF-IDF on training texts; train LightGBM regressor on those features.

    LightGBM is imported lazily so that other stages can run without
    `lightgbm` installed.
    """
    try:
        from lightgbm import LGBMRegressor  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "PS1 requires `lightgbm`. Install with: pip install lightgbm"
        ) from exc

    vec = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
    )
    X_train = vec.fit_transform(train_texts)

    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, train_scores)
    metadata = {
        "tfidf_max_features": vec.max_features,
        "tfidf_ngram_range": list(vec.ngram_range),
        "tfidf_actual_features": X_train.shape[1],
        "lgbm_n_estimators": model.n_estimators,
        "lgbm_learning_rate": model.learning_rate,
        "lgbm_random_state": seed,
    }
    return vec, model, metadata


def main(run_dir: Path, seed: int) -> Path:
    train_path = run_dir / "stage_0" / "essays_train.csv"
    test_path = run_dir / "stage_0" / "essays_test.csv"
    if not (train_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            f"PS0 outputs missing under {run_dir / 'stage_0'}. Run PS0 first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    t0 = time.monotonic()
    vec, model, meta = _train(
        train_texts=train_df["essay_text"].tolist(),
        train_scores=train_df["true_score"].tolist(),
        seed=seed,
    )
    train_seconds = time.monotonic() - t0

    X_test = vec.transform(test_df["essay_text"].tolist())
    predicted = model.predict(X_test)

    predictions = pd.DataFrame(
        {
            "essay_id": test_df["essay_id"].values,
            "essay_text": test_df["essay_text"].values,
            "predicted_score": predicted,
            "true_score": test_df["true_score"].values,
        }
    )

    out_dir = stage_dir(run_dir, "stage_1_aes")
    pred_path = out_dir / "predictions.csv"
    model_path = out_dir / "model.pkl"
    meta_path = out_dir / "train_metadata.json"

    predictions.to_csv(pred_path, index=False)
    joblib.dump({"vectorizer": vec, "model": model}, model_path)
    meta["train_seconds"] = train_seconds
    meta["n_train"] = len(train_df)
    meta["n_test"] = len(test_df)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"PS1 trained on {len(train_df):,} essays in {train_seconds:.1f}s; "
        f"predicted on {len(test_df):,} test essays."
    )
    return pred_path


def cli() -> None:
    parser = argparse.ArgumentParser(description="PS1 — AES baseline (TF-IDF + LightGBM)")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42001)
    args = parser.parse_args()
    run_dir = resolve_or_create_run_dir(seed=-1, explicit=args.run_dir)
    out_path = main(run_dir=run_dir, seed=args.seed)
    print(f"PS1 done. Wrote {out_path}")


if __name__ == "__main__":
    cli()
