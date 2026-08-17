"""Phoneme-sequence corruption (substitute, delete, insert), shared by the robustness probe and noise-augmented training."""

import random
from typing import List, Optional

KINDS = ("substitute", "delete", "insert")

def phoneme_inventory(phoneme_series) -> List[str]:
    """Every distinct phoneme in the corpus, so substitutions and insertions use only seen symbols."""
    return sorted({p for seq in phoneme_series for p in str(seq).split()})

def corrupt(phonemes: str, kind: str, rate: float,
            rng: random.Random, inventory: List[str]) -> str:
    """Corrupt a fraction `rate` of the phonemes in a space-separated ARPAbet string (at least one when rate > 0)."""
    toks = phonemes.split()
    if not toks or rate <= 0:
        return phonemes

    n = max(1, round(len(toks) * rate))
    idxs = rng.sample(range(len(toks)), min(n, len(toks)))

    if kind == "substitute":
        for i in idxs:
            choices = [p for p in inventory if p != toks[i]]
            if choices:
                toks[i] = rng.choice(choices)
    elif kind == "delete":
        for i in sorted(idxs, reverse=True):
            del toks[i]
    elif kind == "insert":
        for i in sorted(idxs, reverse=True):
            toks.insert(i, rng.choice(inventory))
    else:
        raise ValueError(f"unknown corruption kind: {kind}")

    return " ".join(toks)

def corrupt_random(phonemes: str, rng: random.Random, inventory: List[str],
                   prob: float, rate_min: float, rate_max: float,
                   kinds: Optional[tuple] = None) -> str:
    """Training-time corruption: with probability `prob`, apply one random corruption at a rate in [rate_min, rate_max]."""
    if rng.random() >= prob:
        return phonemes
    kind = rng.choice(kinds or KINDS)
    rate = rng.uniform(rate_min, rate_max)
    return corrupt(phonemes, kind, rate, rng, inventory)
