"""
Phoneme-sequence corruption, shared by the inference-time robustness probe
(analysis/noise_probe.py) and noise-augmented training (dryrun.py).

The model is trained on ground-truth ARPAbet phonemes, but in a real lip-reading
pipeline the phonemes arrive from a visual front-end that makes mistakes. These
three corruptions stand in for the three ways that front-end fails:

    substitute  it heard a different phoneme      (visually confusable sounds)
    delete      it missed a phoneme entirely      (fast speech, occlusion)
    insert      it hallucinated an extra phoneme  (jitter, coarticulation)

One implementation used in both places on purpose: the noise the model trains
against and the noise it is evaluated against must be the same function, or the
robustness numbers mean nothing.
"""

import random
from typing import List, Optional

KINDS = ("substitute", "delete", "insert")


def phoneme_inventory(phoneme_series) -> List[str]:
    """Every distinct phoneme in the corpus, so substitutions and insertions only
    ever use symbols the model has actually seen in training."""
    return sorted({p for seq in phoneme_series for p in str(seq).split()})


def corrupt(phonemes: str, kind: str, rate: float,
            rng: random.Random, inventory: List[str]) -> str:
    """Corrupt `rate` of the phonemes in a space-separated ARPAbet string.

    At least one phoneme is always affected when rate > 0, so short sentences
    are not silently left clean by rounding.
    """
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
        # Reverse order so earlier deletions don't shift later indices.
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
    """Training-time corruption: with probability `prob`, apply one randomly
    chosen corruption at a rate drawn from [rate_min, rate_max].

    Mixing rates rather than fixing one stops the model from calibrating to a
    single noise level, and leaving (1 - prob) of examples clean is what keeps
    clean-input performance from regressing.
    """
    if rng.random() >= prob:
        return phonemes
    kind = rng.choice(kinds or KINDS)
    rate = rng.uniform(rate_min, rate_max)
    return corrupt(phonemes, kind, rate, rng, inventory)
