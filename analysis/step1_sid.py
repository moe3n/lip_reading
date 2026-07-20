"""
Step 1 of the error pattern analysis: SID (substitution / insertion / deletion)
breakdown at word and character level, on the dedup beam-5 predictions.

Mirrors the first analysis in the shared P2T framework: before looking at
phonemes, establish which error TYPE dominates. Reuses error_analysis.normalise()
so these counts line up with the WER/CER already reported.

Usage:  python analysis/step1_sid.py
"""

import os
import sys
import pandas as pd
import jiwer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from p2t_lora.evaluation.error_analysis import normalise

PRED_CSV = "p2t_lora_checkpoints_dedup/predictions_beam5.csv"


def report(out, level, ref_unit_total):
    """out = jiwer output object; rates are per reference unit (the WER/CER denominator)."""
    rows = [
        ("Substitutions", out.substitutions),
        ("Insertions",    out.insertions),
        ("Deletions",     out.deletions),
        ("Hits",          out.hits),
    ]
    print(f"\n  {level}-level SID   (denominator = {ref_unit_total} reference {level}s)")
    print(f"  {'Error type':<16}{'Count':>8}{'Rate':>10}")
    print("  " + "-" * 34)
    for label, n in rows:
        print(f"  {label:<16}{n:>8}{n / ref_unit_total * 100:>9.2f}%")
    total_err = out.substitutions + out.insertions + out.deletions
    print(f"  {'Total errors':<16}{total_err:>8}{total_err / ref_unit_total * 100:>9.2f}%")
    # Share of the error mass each type accounts for -- this is the number that
    # says which failure mode dominates, independent of overall error volume.
    print(f"\n  Share of all errors: "
          f"sub {out.substitutions / total_err * 100:.1f}%, "
          f"ins {out.insertions / total_err * 100:.1f}%, "
          f"del {out.deletions / total_err * 100:.1f}%")


def main():
    df = pd.read_csv(PRED_CSV).fillna("")
    refs = [normalise(t) for t in df["target"]]
    hyps = [normalise(p) for p in df["prediction"]]

    print("=" * 60)
    print(f"  Step 1: SID breakdown — {PRED_CSV}")
    print(f"  {len(df)} sentence pairs")
    print("=" * 60)

    w = jiwer.process_words(refs, hyps)
    report(w, "Word", sum(len(r.split()) for r in refs))

    c = jiwer.process_characters(refs, hyps)
    report(c, "Character", sum(len(r) for r in refs))

    # How many sentences are wrong at all, and how wrong -- an average over only
    # the failing rows, since 92% of rows are perfect and would flatten it.
    wrong = [(r, h) for r, h in zip(refs, hyps) if r != h]
    print(f"\n  Sentences: {len(df) - len(wrong)} exact / {len(wrong)} with at least one error")
    if wrong:
        w_wrong = jiwer.process_words([r for r, _ in wrong], [h for _, h in wrong])
        n_ref_words = sum(len(r.split()) for r, _ in wrong)
        print(f"  Within the {len(wrong)} failing sentences: WER {w_wrong.wer * 100:.2f}%, "
              f"{(w_wrong.substitutions + w_wrong.insertions + w_wrong.deletions) / len(wrong):.2f} errors/sentence")
        print(f"  (those {len(wrong)} sentences hold {n_ref_words} reference words)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
