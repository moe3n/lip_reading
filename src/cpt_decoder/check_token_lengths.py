"""
check_token_lengths.py

Standalone diagnostic: tokenizes the FULL LRS2 corpus (48,164 sentences) with
the real Llama 3.2 tokenizer and reports how many tokens our phoneme-prefix
inputs and sentence targets actually take up. This is the evidence behind two
decisions: (1) stripping the literal "<space>" marker out of the phoneme
string before tokenizing it (it has no tokenizer special-token entry, so it
otherwise shatters into 3 sub-tokens per word boundary and nearly doubles
input length for zero phonetic information), and (2) the max_input_len /
max_target_len values to use for the real Llama 3.2:3B run on the uni PC.

Run it from the repo root (same convention as dryrun.py):
    python3 -m src.cpt_decoder.check_token_lengths

Output is plain text -- meant to be screenshotted as-is for a supervisor
update.

Tokenizer note
--------------
meta-llama/Llama-3.2-3B is a GATED repo on Hugging Face -- it needs an
accepted license + access token to download. This script instead loads
"unsloth/Llama-3.2-3B", an UNGATED re-upload that ships the identical
tokenizer.json / merges as Meta's official release (same 128,000-token
vocabulary), so the token counts below are exactly what the real model
will see. Once HF gated access is sorted on the uni PC, you can switch
TOKENIZER_NAME below to "meta-llama/Llama-3.2-3B" -- the numbers will not
change, since it's the same tokenizer either way.
"""

import os
import sys
import statistics as st

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpt_decoder.data.loader import load_original_phoneme_text_pairs  # noqa: E402

TOKENIZER_NAME = "unsloth/Llama-3.2-3B"  # tokenizer-identical mirror of meta-llama/Llama-3.2-3B
CURRENT_MAX_INPUT_LEN = 96   # current dryrun.py CFG value (phoneme prefix budget)
CURRENT_MAX_TARGET_LEN = 32  # current dryrun.py CFG value (sentence target budget)


def strip_space_marker(phonemes_cleaned: str) -> str:
    """The proposed fix: also remove the literal '<space>' marker token."""
    return phonemes_cleaned.replace("<space>", " ")


def summarize(label: str, lengths: list[int], budget: int) -> None:
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    p95 = lengths_sorted[int(0.95 * n)]
    over_budget = sum(1 for l in lengths if l > budget)
    print(
        f"  {label:42s} mean={st.mean(lengths):6.1f}  median={st.median(lengths):5}  "
        f"p95={p95:5}  max={max(lengths):5}   ->  {100 * over_budget / n:5.2f}% exceed budget of {budget}"
    )


def main() -> None:
    print("=" * 78)
    print("  Token-length audit -- real Llama 3.2 tokenizer, full LRS2 corpus")
    print("=" * 78)

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"  Tokenizer     : {TOKENIZER_NAME}  (vocab size {tok.vocab_size})")

    df = load_original_phoneme_text_pairs()
    print(f"  Corpus rows   : {len(df)}")
    print()

    phon_current = [len(tok(p)["input_ids"]) for p in df["phonemes"]]
    phon_fixed = [len(tok(strip_space_marker(p))["input_ids"]) for p in df["phonemes"]]
    sent_lens = [len(tok(s)["input_ids"]) for s in df["sentence"]]

    print("  Phoneme prefix (model INPUT):")
    summarize("current clean_phoneme_seq()", phon_current, CURRENT_MAX_INPUT_LEN)
    summarize("+ <space> marker stripped (proposed fix)", phon_fixed, CURRENT_MAX_INPUT_LEN)
    print()
    print("  Sentence (model TARGET):")
    summarize("clean_sentence()", sent_lens, CURRENT_MAX_TARGET_LEN)
    print()

    print("-" * 78)
    print("  Recommendation:")
    print(f"    max_input_len  = {CURRENT_MAX_INPUT_LEN}  (after the <space> fix, covers "
          f"{100 * sum(1 for l in phon_fixed if l <= CURRENT_MAX_INPUT_LEN) / len(phon_fixed):.1f}% of the corpus with no truncation)")
    print(f"    max_target_len = {CURRENT_MAX_TARGET_LEN}  (covers "
          f"{100 * sum(1 for l in sent_lens if l <= CURRENT_MAX_TARGET_LEN) / len(sent_lens):.1f}% of sentences with no truncation)")
    print("=" * 78)


if __name__ == "__main__":
    main()
