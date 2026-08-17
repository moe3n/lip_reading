"""
Re-run Stage 2 (substitution-category breakdown) on the preserved 48,164-row
full-corpus zero-shot predictions, stratified by Homophone / Non-Homophone
subset as recorded in view_full_48164.txt.

Inputs:
  results/preds_full_48164.jsonl  : 48,164 {index, phonemes_raw, target, prediction}
  view_full_48164.txt             : 48,164 lines 'STATUS | HOMO | PREDICTED | TARGET'
                                    (in the same order as preds_full_48164.jsonl —
                                    same index, same order)

Output:
  results/stage2_full_48164.json  : structured report identical in shape to
                                    the 45,839-row report's Stage 2 / Stage 3
                                    tables, but for the full 48,164 rows.

Run:
  .venv\\Scripts\\python.exe zero-shot\\baseline_with_full_corpus\\re_analyse_full.py
"""

import json
import os
import sys
from collections import Counter

# Make src/cpt_decoder importable so we can reuse the project's Stage 2
# classifier without copy-pasting it.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cpt_decoder.evaluation.error_analysis import error_category_report

HERE = os.path.dirname(os.path.abspath(__file__))


def load_view_homo_mask(view_path):
    """Read view_full_48164.txt and return a list[bool], one per row."""
    mask = []
    with open(view_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # header
            line = line.rstrip("\n")
            # Format: "WRONG | H | <pred> | <target>"  or  "OK    | - | <pred> | <target>"
            parts = line.split(" | ", 3)
            if len(parts) < 4:
                continue
            mask.append(parts[1].strip() == "H")
    return mask


def main():
    preds_path = os.path.join(HERE, "results", "preds_full_48164.jsonl")
    view_path  = os.path.join(HERE, "view_full_48164.txt")
    out_path   = os.path.join(HERE, "results", "stage2_full_48164.json")

    print(f"Loading {preds_path} ...")
    refs, hyps = [], []
    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            refs.append(row["target"])
            hyps.append(row["prediction"])

    print(f"Loading homophone mask from {view_path} ...")
    mask = load_view_homo_mask(view_path)

    assert len(refs) == len(hyps) == len(mask), (
        f"row counts disagree: refs={len(refs)}, hyps={len(hyps)}, mask={len(mask)}"
    )
    print(f"Loaded {len(refs)} rows "
          f"(homophone={sum(mask)}, non-homophone={sum(not m for m in mask)}).")

    print("Running error_category_report(use_llm=False) ...")
    print("  (Stage 3 grammar-only; near-homophone lookup brute-scans the CMU dict,")
    print("   so this will take several minutes on 48,164 rows.)")

    report = error_category_report(
        all_refs=refs,
        all_hyps=hyps,
        homo_mask=mask,
        use_llm=False,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport written to {out_path}\n")
    print("─── Headline ───")
    for label in ("overall", "homophone", "non_homophone"):
        r = report.get(label, {})
        totals = r.get("totals", {})
        cats = r.get("substitution_categories", {})
        n_sub = totals.get("n_substitutions", 0)
        n_h = cats.get("Homophone", 0)
        n_nh = cats.get("Near-homophone", 0)
        n_o = cats.get("Other", 0)
        pct = lambda x: (100.0 * x / n_sub) if n_sub else 0.0
        print(f"  {label:>14s}: subs={n_sub:>7,d}  "
              f"H={n_h:>6,d} ({pct(n_h):4.1f}%)  "
              f"NH={n_nh:>6,d} ({pct(n_nh):4.1f}%)  "
              f"O={n_o:>7,d} ({pct(n_o):4.1f}%)")


if __name__ == "__main__":
    main()
