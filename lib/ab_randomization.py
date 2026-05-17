"""A/B order randomization for LLM pairwise calls.

Lihi playbook step 10 requires randomizing which essay appears as A and
which as B in each prompt, to neutralize the position bias of LLM
judges. The randomization is seeded so a rerun with the same master
seed reproduces the exact sequence of prompts and the exact win-matrix
update order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class PairOrder:
    """One ordered pair to send to the LLM.

    `slot_a` and `slot_b` are the item_ids that fill the {essay_a} and
    {essay_b} placeholders in the prompt. After the LLM returns "A" or
    "B", the win is credited to `slot_a` or `slot_b` respectively. The
    win matrix update is symmetric in (i, j) regardless of orientation;
    the orientation only matters for the LLM prompt itself.
    """

    item_i: Hashable
    item_j: Hashable
    slot_a: Hashable
    slot_b: Hashable

    @property
    def orientation(self) -> str:
        """Returns "AB" if slot_a==item_i else "BA"."""
        return "AB" if self.slot_a == self.item_i else "BA"


def generate_pair_orderings(
    item_ids: Sequence[Hashable],
    seed: int,
) -> list[PairOrder]:
    """Build the full ordered list of pairs for one top-k subset.

    For each unordered pair (i, j) with i < j in the input order, flip
    a coin (seeded) to decide which item fills slot A. Returns a list
    of length C(k, 2) preserving the input order over (i, j). Half of
    the pairs (in expectation) get orientation "AB", half "BA".

    Parameters
    ----------
    item_ids
        The top-k items in any deterministic order (the caller
        typically passes them in rank order from the AES baseline).
    seed
        Random seed for the per-pair orientation coin flips.

    Returns
    -------
    list[PairOrder] of length k*(k-1)/2.
    """
    rng = np.random.default_rng(seed)
    orderings: list[PairOrder] = []
    ids = list(item_ids)
    for idx_i in range(len(ids)):
        for idx_j in range(idx_i + 1, len(ids)):
            i = ids[idx_i]
            j = ids[idx_j]
            if rng.integers(0, 2) == 0:
                slot_a, slot_b = i, j
            else:
                slot_a, slot_b = j, i
            orderings.append(PairOrder(item_i=i, item_j=j, slot_a=slot_a, slot_b=slot_b))
    return orderings


def winner_to_item_id(pair_order: PairOrder, llm_letter: str) -> Hashable:
    """Map the LLM's "A"/"B" answer back to the actual item_id that won.

    Raises ValueError on any letter other than "A" or "B".
    """
    letter = llm_letter.strip().upper()
    if letter == "A":
        return pair_order.slot_a
    if letter == "B":
        return pair_order.slot_b
    raise ValueError(f"Unexpected LLM letter: {llm_letter!r}")
