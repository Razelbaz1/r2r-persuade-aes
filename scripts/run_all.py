"""Orchestrator — run the full PERSUADE pipeline end-to-end.

Resolves a fresh run-dir, runs PS0..PS5 in sequence, then writes the
run summary into `manifest.yaml`. Each stage is callable standalone too
(useful when iterating on one stage without re-running expensive ones
upstream); this orchestrator is the convenience wrapper.

Usage
-----
    python scripts/run_all.py --seed 42 --provider anthropic
    python scripts/run_all.py --seed 42 --provider openai --model gpt-4o-2024-08-06
    python scripts/run_all.py --seed 42 --provider anthropic --skip-download
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import sys
from pathlib import Path

import yaml

from _common import (
    PACKAGE_ROOT,
    derive_seeds,
    resolve_or_create_run_dir,
)


def _load_stage(module_name: str):
    """Import a stage module from the scripts package (which is this dir)."""
    if str(PACKAGE_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))
    return importlib.import_module(module_name)


def _update_manifest(
    run_dir: Path,
    seed: int,
    provider: str,
    model_id: str,
    temperature: float,
    top_k: int,
) -> None:
    manifest_path = PACKAGE_ROOT / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["last_run"] = {
        "run_dir": str(run_dir.relative_to(PACKAGE_ROOT)).replace("\\", "/"),
        "seed": int(seed),
        "provider": provider,
        "model_id": model_id,
        "temperature": float(temperature),
        "top_k": int(top_k),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def cli() -> None:
    parser = argparse.ArgumentParser(description="PERSUADE full pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic"),
        required=True,
        help="LLM provider for PS3 pairwise judgments",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id (defaults: gpt-4o-2024-08-06 for openai, claude-sonnet-4-6 for anthropic)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Optional cap on LLM API calls in PS3 (for cost-controlled smoke tests)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip PS0 if the dataset is already on disk",
    )
    parser.add_argument(
        "--skip-aes",
        action="store_true",
        help=(
            "Skip PS1 if `<run-dir>/stage_1_aes/predictions.csv` already exists. "
            "Use when iterating on PS3+ with a frozen AES baseline."
        ),
    )
    args = parser.parse_args()

    seeds = derive_seeds(args.seed)
    run_dir = resolve_or_create_run_dir(seed=args.seed)
    print(f"Run dir: {run_dir}")

    # PS0 — Download + split
    if not args.skip_download:
        ps0 = _load_stage("stage_0_download")
        ps0.main(run_dir=run_dir, seed=seeds["seed_split"])
        print("PS0 done.")
    else:
        print("PS0 skipped (--skip-download).")

    # PS1 — AES baseline via datafan07's Kaggle notebook
    # (stage_1_train_aes_baseline.py is deprecated -- kept in tree for
    # reference but not invoked here.)
    predictions_path = run_dir / "stage_1_aes" / "predictions.csv"
    if args.skip_aes and predictions_path.exists():
        print(f"PS1 skipped (predictions exist at {predictions_path}).")
    else:
        ps1 = _load_stage("stage_1_run_kaggle_notebook")
        ps1.main(run_dir=run_dir)
        print("PS1 done.")

    # PS2 — Top-k selection
    ps2 = _load_stage("stage_2_select_top_k")
    ps2.main(run_dir=run_dir, rank_from=1, rank_to=args.top_k)
    print("PS2 done.")

    # PS3 — LLM pairwise
    ps3 = _load_stage("stage_3_run_llm_pairwise")
    resolved_model = ps3.main(
        run_dir=run_dir,
        provider=args.provider,
        model_id=args.model,
        temperature=args.temperature,
        seed_ab_order=seeds["seed_ab_order"],
        max_calls=args.max_calls,
    )
    print("PS3 done.")

    # PS4 — Copeland fusion
    ps4 = _load_stage("stage_4_fuse_copeland")
    ps4.main(run_dir=run_dir)
    print("PS4 done.")

    # PS5 — Evaluation
    ps5 = _load_stage("stage_5_evaluate")
    ps5.main(run_dir=run_dir)
    print("PS5 done.")

    _update_manifest(
        run_dir=run_dir,
        seed=args.seed,
        provider=args.provider,
        model_id=resolved_model,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f"manifest.yaml updated. Run complete: {run_dir}")


if __name__ == "__main__":
    cli()
