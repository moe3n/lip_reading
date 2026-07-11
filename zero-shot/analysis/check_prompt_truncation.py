"""
Prompt token-length / truncation-budget check for run_baseline.py.

Standalone diagnostic (tokenizer only — no model load, no GPU). Reuses
run_baseline.py's own MODEL/MODES/INSTRUCTIONS/PHONEME_COLUMN/MAX_INPUT_LEN
constants directly rather than duplicating them, so this can never drift out
of sync with what a real run actually does.

Origin: this caught the bug behind the 2026-07-10 5,000-train-row run where
"raw" mode's EM was 0% across the board — MAX_INPUT_LEN=96 (sized for
"clean" phonemes only) was truncating away the entire "Phonemes: ...\nText:"
cue on every raw prompt (truncation_side defaulted to "right"), so the model
had nothing to condition on and collapsed to one fixed unconditional
completion regardless of input. See results/2026-07-10_pre-fix.txt for the
numbers that diagnosed it, and run_baseline.py's MAX_INPUT_LEN comment for
the fix (320 + truncation_side="left"). Re-run this after any instruction-
text or MAX_INPUT_LEN change to confirm the "Text:" cue still survives.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))      # zero-shot/analysis/
_ZS = os.path.dirname(_HERE)                              # zero-shot/
sys.path.insert(0, _ZS)

from transformers import AutoTokenizer                    # noqa: E402
import run_baseline as rb                                 # noqa: E402
from p2t_lora.data import loader as data_loader         # noqa: E402


def main():
    tok = AutoTokenizer.from_pretrained(rb.MODEL)
    df = data_loader.load_original_phoneme_text_pairs()

    lines = [f"MAX_INPUT_LEN = {rb.MAX_INPUT_LEN}", f"corpus rows = {len(df)}", ""]
    for mode in rb.MODES:
        col = rb.PHONEME_COLUMN[mode]
        instr = rb.INSTRUCTIONS[mode]
        lens = sorted(
            len(tok(f"{instr}\n\nPhonemes: {p}\nText:")["input_ids"])
            for p in df[col]
        )
        n = len(lens)
        over = sum(1 for l in lens if l > rb.MAX_INPUT_LEN)
        line = (f"{mode:<6} n={n}  mean={sum(lens) / n:.1f}  "
                f"p99={lens[int(n * 0.99)]}  max={lens[-1]}  "
                f"over_budget={over} ({over / n * 100:.1f}%)")
        print(line)
        lines.append(line)

    out_path = os.path.join(_HERE, "prompt_length_report.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
