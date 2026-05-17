"""PS3 — LLM pairwise judgments on the top-k essays.

For every unordered pair (i, j) in the top-k from PS2, sends the
configured LLM a prompt containing both essays in a randomized A/B
order (seeded by `seed_ab_order` from the master seed). Returns a
binary win, builds a k×k integer win matrix, and writes a per-call
audit log.

Cache strategy
--------------
Every (provider, model, temperature, prompt-hash) triple is cached on
disk under `experiments/persuade_aes/cache/llm_responses.sqlite`. A
rerun with identical inputs replays from cache at zero API cost. The
cache survives `runs/run_NNN_*` deletion — the LLM responses are
expensive and worth preserving across run-dirs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from _common import (
    CACHE_ROOT,
    add_package_to_syspath,
    resolve_or_create_run_dir,
    stage_dir,
)

add_package_to_syspath()
from lib.ab_randomization import generate_pair_orderings, winner_to_item_id  # noqa: E402
from lib.llm_judge import make_judge  # noqa: E402
from lib.prompts import PAIRWISE_SYSTEM_MESSAGE, PAIRWISE_USER_TEMPLATE  # noqa: E402


def main(
    run_dir: Path,
    provider: str,
    model_id: Optional[str],
    temperature: float,
    seed_ab_order: int,
    max_calls: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    top_k_path = run_dir / "stage_2_top_k" / "top_k.csv"
    if not top_k_path.exists():
        raise FileNotFoundError(f"Missing PS2 output: {top_k_path}. Run PS2 first.")
    top_k = pd.read_csv(top_k_path)

    # essay_id is a hex string in PERSUADE (e.g. "1b1975b") -- keep as str.
    essay_lookup: dict[str, str] = {
        str(row.essay_id): row.essay_text for row in top_k.itertuples(index=False)
    }
    item_ids: list[str] = [str(eid) for eid in top_k["essay_id"].tolist()]

    pair_orderings = generate_pair_orderings(item_ids=item_ids, seed=seed_ab_order)
    if max_calls is not None:
        pair_orderings = pair_orderings[:max_calls]
        print(f"Capping LLM calls at {max_calls} (out of {len(item_ids) * (len(item_ids) - 1) // 2}).")

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    judge = make_judge(
        provider=provider,
        model_id=model_id,
        temperature=temperature,
        cache_dir=CACHE_ROOT,
        reasoning_effort=reasoning_effort,
    )
    resolved_model = judge.model_id
    effort_tag = f", reasoning_effort={reasoning_effort}" if reasoning_effort else ""
    print(
        f"Judging {len(pair_orderings)} pairs with provider={provider}, "
        f"model={resolved_model}, temperature={temperature}{effort_tag}"
    )

    W = pd.DataFrame(
        np.zeros((len(item_ids), len(item_ids)), dtype=int),
        index=item_ids,
        columns=item_ids,
    )

    log_rows = []
    for idx, pair_order in enumerate(pair_orderings, start=1):
        user_content = PAIRWISE_USER_TEMPLATE.format(
            essay_a=essay_lookup[str(pair_order.slot_a)],
            essay_b=essay_lookup[str(pair_order.slot_b)],
        )
        response = judge.judge(system=PAIRWISE_SYSTEM_MESSAGE, user=user_content)
        winner = str(winner_to_item_id(pair_order, response.winner))
        loser = str(pair_order.item_j) if winner == str(pair_order.item_i) else str(pair_order.item_i)
        W.loc[winner, loser] += 1

        log_rows.append(
            {
                "pair_index": idx,
                "item_i": str(pair_order.item_i),
                "item_j": str(pair_order.item_j),
                "slot_a": str(pair_order.slot_a),
                "slot_b": str(pair_order.slot_b),
                "orientation": pair_order.orientation,
                "llm_winner": response.winner,
                "winner_item_id": winner,
                "loser_item_id": loser,
                "cached": response.cached,
                "elapsed_seconds": response.elapsed_seconds,
                "raw_response": response.raw,
            }
        )
        cache_tag = "cache" if response.cached else f"{response.elapsed_seconds:.2f}s"
        print(
            f"  [{idx:>3}/{len(pair_orderings)}] "
            f"({pair_order.item_i},{pair_order.item_j}) "
            f"orient={pair_order.orientation} -> {response.winner} (winner={winner}) [{cache_tag}]"
        )

    out_dir = stage_dir(run_dir, "stage_3_llm")
    matrix_path = out_dir / "win_matrix.csv"
    W.to_csv(matrix_path)

    log_path = out_dir / "pairwise_log.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    n_cached = sum(1 for r in log_rows if r["cached"])
    print(
        f"PS3 done. Wrote {matrix_path} and {log_path}. "
        f"Cache hits: {n_cached}/{len(log_rows)}."
    )
    return resolved_model


def cli() -> None:
    parser = argparse.ArgumentParser(description="PS3 — LLM pairwise judgments")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("openai", "anthropic"), required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--seed-ab-order",
        type=int,
        default=421,
        help="Seed for the per-pair A/B coin flip (default 421 = derive_seeds(42)['seed_ab_order'])",
    )
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default=None,
        help="Only acted on by OpenAI reasoning-family models "
             "(gpt-5, o-series). Legacy models accept the flag but "
             "ignore it. Defaults to None (model picks its own default; "
             "the in-code minimal preset still applies as a fallback).",
    )
    args = parser.parse_args()
    run_dir = resolve_or_create_run_dir(seed=-1, explicit=args.run_dir)
    main(
        run_dir=run_dir,
        provider=args.provider,
        model_id=args.model,
        temperature=args.temperature,
        seed_ab_order=args.seed_ab_order,
        max_calls=args.max_calls,
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    cli()
