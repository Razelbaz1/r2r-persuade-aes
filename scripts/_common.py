"""Shared helpers for the PERSUADE pipeline scripts.

Centralizes path resolution, run-dir layout, and import-path setup so
each numbered stage can be a thin script that calls into `lib/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Package root: experiments/persuade_aes/
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PACKAGE_ROOT / "runs"
CACHE_ROOT = PACKAGE_ROOT / "cache"

# Stage subdir names under <run-dir>/.
STAGE_DIRS = {
    "stage_0": "stage_0",
    "stage_1_aes": "stage_1_aes",
    "stage_2_top_k": "stage_2_top_k",
    "stage_3_llm": "stage_3_llm",
    "stage_4_fusion": "stage_4_fusion",
    "stage_5_eval": "stage_5_eval",
}


def add_package_to_syspath() -> None:
    """Add the package root to sys.path so `import lib.*` works in scripts."""
    root_str = str(PACKAGE_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def stage_dir(run_dir: Path, stage_key: str) -> Path:
    """Return (and create) `<run-dir>/<stage_key>/`."""
    if stage_key not in STAGE_DIRS:
        raise KeyError(f"Unknown stage key: {stage_key!r}. Known: {list(STAGE_DIRS)}")
    out = run_dir / STAGE_DIRS[stage_key]
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_or_create_run_dir(seed: int, explicit: Path | None = None) -> Path:
    """Return an existing run-dir or allocate the next available one.

    Naming convention matches the Sushi pipeline:
    `runs/run_NNN_seed_YY/` with NNN auto-incremented zero-padded.

    If `explicit` is provided, return it as-is (the script is being run
    in single-stage mode against an existing run-dir).
    """
    if explicit is not None:
        explicit = explicit.resolve()
        if not explicit.exists():
            explicit.mkdir(parents=True, exist_ok=True)
        return explicit
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in RUNS_ROOT.glob("run_*_seed_*"))
    next_idx = 1
    if existing:
        last = existing[-1]
        try:
            next_idx = int(last.split("_")[1]) + 1
        except (IndexError, ValueError):
            next_idx = len(existing) + 1
    run_dir = RUNS_ROOT / f"run_{next_idx:03d}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def derive_seeds(master_seed: int) -> dict[str, int]:
    """Derive per-stage seeds from the master seed.

    Mirrors `pipeline/run_pipeline.py:derive_seeds` so future
    side-by-side runs share a discoverable seed scheme.
    """
    return {
        "seed_split": master_seed,
        "seed_aes": master_seed * 1000 + 1,
        "seed_ab_order": master_seed * 10 + 1,
    }
