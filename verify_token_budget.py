"""
Verify the token-length budget for the CPT Decoder.

Runs the full 48,164-row LRS2 corpus through the real Llama 3.2 tokenizer
and reports input/target length distributions, both before and after the
proposed <space>-marker fix.

No GPU, no model weights, no HF gated access required — uses the
unsloth/Llama-3.2-3B mirror, which ships a byte-identical tokenizer.json
to the gated meta-llama/Llama-3.2-3B release.
"""

import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = "data/LRS2 Sentences with Phoneme Sequences/sentphonemepairs_LRS2_original.csv"
TOKENIZER_NAME = "unsloth/Llama-3.2-3B"   # ungated, tokenizer byte-identical to gated release
INPUT_BUDGET = 98
TARGET_BUDGET = 34


# ── Cleaning functions (mirror data/loader.py) ────────────────────────────────
def clean_phoneme_seq_current(seq: str) -> str:
    """Current cleaning in data/loader.py — leaves <space> intact."""
    seq = seq.strip()
    seq = re.sub(r"<SOS>|<EOS>", "", seq)
    seq = re.sub(r"[012]", "", seq)
    seq = re.sub(r"\s+", " ", seq).strip()
    return seq


def clean_phoneme_seq_fixed(seq: str) -> str:
    """Proposed fix — also strips the <space> word-boundary marker."""
    seq = clean_phoneme_seq_current(seq)
    seq = seq.replace("<space>", "|")  # replace with a single-character word boundary
    seq = re.sub(r"\s+", " ", seq).strip()
    return seq


def clean_sentence(text: str) -> str:
    return text.strip().upper()


# ── Stats helper ──────────────────────────────────────────────────────────────
def report(name: str, lengths: np.ndarray, budget: int) -> None:
    over = (lengths > budget).sum()
    pct_over = 100 * over / len(lengths)
    print(
        f"  {name:<45s}"
        f"mean={lengths.mean():>5.1f}  "
        f"median={np.median(lengths):>5.1f}  "
        f"p95={np.percentile(lengths, 95):>5.0f}  "
        f"max={lengths.max():>4.0f}  "
        f"-> {pct_over:>5.2f}% exceed budget of {budget}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"  vocab size: {tok.vocab_size:,}\n")

    print(f"Loading corpus: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, header=None, names=["sentence", "phonemes_raw"])
    df = df.dropna().reset_index(drop=True)
    print(f"  rows: {len(df):,}\n")

    # Apply both cleaning variants
    phon_current = df["phonemes_raw"].apply(clean_phoneme_seq_current).tolist()
    phon_fixed   = df["phonemes_raw"].apply(clean_phoneme_seq_fixed).tolist()
    sentences    = df["sentence"].apply(clean_sentence).tolist()

    # Tokenise (no special tokens — we want raw subword counts)
    print("Tokenising...")
    lens_current = np.array([len(tok.encode(s, add_special_tokens=False)) for s in phon_current])
    lens_fixed   = np.array([len(tok.encode(s, add_special_tokens=False)) for s in phon_fixed])
    lens_target  = np.array([len(tok.encode(s, add_special_tokens=False)) for s in sentences])

    # Report
    print("\nPhoneme prefix (model input):")
    report("current clean phoneme with <space> marker ",              lens_current, INPUT_BUDGET)
    report("<space> marker stripped", lens_fixed,   INPUT_BUDGET)

    print("\nSentence (model output):")
    report("clean sentence ", lens_target, TARGET_BUDGET)

    # # Single-token <space> check — confirms the bloat mechanism
    # print("\nSanity check — how the literal '<space>' string tokenises:")
    # pieces = tok.tokenize("<space>")
    # print(f"  '<space>' -> {pieces}  ({len(pieces)} sub-tokens)")

    print("\n:")
    pct_in = 100 * (lens_fixed <= INPUT_BUDGET).sum() / len(lens_fixed)
    pct_tg = 100 * (lens_target <= TARGET_BUDGET).sum() / len(lens_target)
    print(f"  max_input_len  = {INPUT_BUDGET}  (covers {pct_in:.3f}% of corpus)")
    print(f"  max_target_len = {TARGET_BUDGET}  (covers {pct_tg:.3f}% of sentences)")


if __name__ == "__main__":
    main()