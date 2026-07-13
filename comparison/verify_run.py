"""
Leakage / hygiene audit for a fine-tuned run's validation predictions.

Read-only reproduction of VERIFICATION_AUDIT_2026-07-12.md, saved as code so it
can be re-run on any run's predictions.csv (not just the one audited by hand).
Produces per-row leakage labels with corpus row numbers as the reference, plus
a stratified metrics summary that separates exact-duplicate, near-duplicate, and
truly-novel validation sentences.

Why this matters: the LRS2 corpus has repeated sentences (48,164 rows, 45,455
unique), so a sequential train/val split can leak a sentence verbatim across the
boundary even though no ROW is shared. This script quantifies that and reports
the honest, duplicate-free number alongside the headline.

Usage:
    python -m comparison.verify_run                     # audits p2t_lora_checkpoints_full/
    CPT_VERIFY_PREDICTIONS=some/predictions.csv python -m comparison.verify_run

Split boundaries (TRAIN_N / VAL_N) match src/p2t_lora/dryrun.py's CPT_SEQ_SPLIT
and zero-shot/run_baseline.py — the last TEST_N rows are never touched here.
"""

import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import jiwer                                                    # noqa: E402
import pandas as pd                                            # noqa: E402

from p2t_lora.data import loader as data_loader                # noqa: E402

TRAIN_N, VAL_N, TEST_N = 45839, 1082, 1243   # keep in sync with dryrun.py / run_baseline.py

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PREDICTIONS = os.environ.get(
    "CPT_VERIFY_PREDICTIONS",
    os.path.join(REPO, "p2t_lora_checkpoints_full", "predictions.csv"),
)
OUT_DIR = os.path.join(HERE, "results", "verification")


def norm(t):
    t = str(t).lower()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def deletion_variants(words):
    """All strings formed by deleting exactly one word — the hashing trick for
    finding word-edit-distance-1 neighbours without an O(n^2) pairwise scan."""
    return {" ".join(words[:i] + words[i + 1:]) for i in range(len(words))}


def wer_em(refs_n, hyps_n):
    if not refs_n:
        return None, None
    wer = jiwer.wer(refs_n, hyps_n) * 100
    em = sum(r == h for r, h in zip(refs_n, hyps_n)) / len(refs_n) * 100
    return wer, em


def main():
    print(f"Auditing: {PREDICTIONS}")
    corpus = data_loader.load_original_phoneme_text_pairs()
    train = corpus.iloc[:TRAIN_N]
    val = corpus.iloc[TRAIN_N:TRAIN_N + VAL_N].reset_index(drop=True)
    preds = pd.read_csv(PREDICTIONS)

    if len(preds) != len(val):
        print(f"  WARNING: predictions has {len(preds)} rows, val slice has {len(val)}")
    aligned = (preds["target"].reset_index(drop=True) == val["sentence"].reset_index(drop=True)).all()
    print(f"  row-aligned with corpus val slice: {aligned}")

    # sentence -> list of 0-based corpus row numbers it occupies in TRAIN
    train_rows = {}
    for i, s in enumerate(train["sentence"]):
        train_rows.setdefault(s, []).append(i)
    train_set = set(train_rows)

    # deletion-1 variants of every train sentence, for the near-duplicate screen
    train_del = set()
    for s in train_set:
        train_del |= deletion_variants(s.split())

    refs_n = [norm(t) for t in preds["target"]]
    hyps_n = [norm(t) for t in preds["prediction"].fillna("")]

    rows = []
    for i in range(len(val)):
        sent = val["sentence"].iloc[i]
        corpus_row = TRAIN_N + i                       # 0-based position in the full corpus
        words = sent.split()
        d1 = deletion_variants(words)

        exact_matches = train_rows.get(sent, [])
        if exact_matches:
            category = "exact_dup"
        elif (sent in train_del) or (d1 & train_set) or (d1 & train_del):
            category = "near_dup"    # within ~2 word-edits (incl. one substitution)
        else:
            category = "novel"

        rows.append({
            "val_index": i,
            "corpus_row": corpus_row,
            "category": category,
            "correct": refs_n[i] == hyps_n[i],
            "is_homophone": bool(preds["is_homophone"].iloc[i]) if "is_homophone" in preds.columns else "",
            "n_train_exact_matches": len(exact_matches),
            "train_match_rows": ";".join(map(str, exact_matches[:20])),  # cap the list width
            "phonemes": val["phonemes"].iloc[i],
            "target": sent,
            "prediction": preds["prediction"].iloc[i],
        })

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── per-row leakage labels ────────────────────────────────────────────
    per_row_path = os.path.join(OUT_DIR, "leakage_per_row.csv")
    fieldnames = ["val_index", "corpus_row", "category", "correct", "is_homophone",
                  "n_train_exact_matches", "train_match_rows", "phonemes", "target", "prediction"]
    with open(per_row_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # ── separate files per leakage category, for easy reference ───────────
    for cat in ("exact_dup", "near_dup", "novel"):
        cat_rows = [r for r in rows if r["category"] == cat]
        with open(os.path.join(OUT_DIR, f"rows_{cat}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(cat_rows)

    # ── stratified metrics summary ────────────────────────────────────────
    summary = []
    def add(label, mask):
        r = [x for x, m in zip(refs_n, mask) if m]
        h = [x for x, m in zip(hyps_n, mask) if m]
        wer, em = wer_em(r, h)
        if wer is not None:
            summary.append({"subset": label, "n": len(r),
                            "WER": round(wer, 4), "ExactMatch": round(em, 4)})

    cats = [r["category"] for r in rows]
    add("Headline (all val rows)", [True] * len(rows))
    add("Exact duplicate of train", [c == "exact_dup" for c in cats])
    add("Near duplicate (<=2 edits)", [c == "near_dup" for c in cats])
    add("Truly novel (no exact/near dup)", [c == "novel" for c in cats])

    summary_path = os.path.join(OUT_DIR, "leakage_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "n", "WER", "ExactMatch"])
        w.writeheader()
        w.writerows(summary)

    # ── machine-readable rollup ───────────────────────────────────────────
    counts = {c: cats.count(c) for c in ("exact_dup", "near_dup", "novel")}
    rollup = {
        "predictions_file": os.path.relpath(PREDICTIONS, REPO),
        "n_val": len(rows),
        "row_aligned": bool(aligned),
        "category_counts": counts,
        "category_pct": {k: round(v / len(rows) * 100, 2) for k, v in counts.items()},
        "summary": summary,
    }
    with open(os.path.join(OUT_DIR, "leakage_rollup.json"), "w", encoding="utf-8") as f:
        json.dump(rollup, f, indent=2)

    print(f"\ncategory counts: {counts}")
    print("\nstratified metrics:")
    for s in summary:
        print(f"  {s['subset']:<34} n={s['n']:>4}  WER={s['WER']:>7.4f}%  EM={s['ExactMatch']:>7.4f}%")
    print(f"\nSaved to {OUT_DIR}/")
    print("  leakage_per_row.csv          — every val row labelled, with corpus_row + train_match_rows")
    print("  rows_exact_dup.csv / rows_near_dup.csv / rows_novel.csv — the three groups split out")
    print("  leakage_summary.csv          — stratified WER/EM table")
    print("  leakage_rollup.json          — machine-readable rollup")


if __name__ == "__main__":
    main()
