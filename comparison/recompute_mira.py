"""
Recompute Mira Fleite's existing zero-shot baseline JSON through OUR OWN
metrics (metrics.py + extended_metrics.py) instead of trusting her reported
numbers directly -- removes metric-implementation differences as a confound
before comparing her prompting-only (Llama 3.1 8B) results against ours.

Source: "Other student's work/mira's/ZeroShot_baseline_results_45840_samples.json"
(45,839 rows). Her row 0 is CSV row 1 of sentphonemepairs_LRS2_original.csv,
NOT row 0 -- a documented off-by-one (her run silently skipped CSV row 0),
corrected here before scoring.

Scope: WER/CER/PER/BLEU/EM (stratified overall/homophone/non-homophone), SID
breakdown, AER, and heuristic+panphon WPER, over the full 45,839 rows -- all
fast, no model calls. Grammar (needs local Java, unavailable in this
environment) and semantic similarity (BERTScore over 45,839 pairs is a real
compute commitment) are deliberately NOT run here -- see comparison/README.md
for how to run those separately when wanted.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from p2t_lora.data import loader as data_loader                              # noqa: E402
from p2t_lora.evaluation.metrics import stratified_evaluate, print_results   # noqa: E402
from p2t_lora.evaluation import extended_metrics as ext                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MIRA_JSON = os.path.join(
    os.path.dirname(HERE), "Other student's work", "mira's",
    "ZeroShot_baseline_results_45840_samples.json",
)
OUT_DIR = os.path.join(HERE, "results")


def main():
    with open(MIRA_JSON, "r", encoding="utf-8") as f:
        mira = json.load(f)["results"]
    print(f"Loaded {len(mira)} rows from Mira's baseline JSON.")

    # Her index i == our CSV row i+1 (documented off-by-one).
    corpus = data_loader.load_original_phoneme_text_pairs()
    homo_set = set(data_loader.load_homophone_sentences()["sentence"])

    refs, hyps, is_homo = [], [], []
    mismatches = 0
    for row in mira:
        csv_idx = row["index"] + 1
        our_sentence = corpus.iloc[csv_idx]["sentence"]
        her_expected = row["expected_text"].strip().upper()
        if our_sentence != her_expected:
            mismatches += 1
        refs.append(our_sentence)
        hyps.append((row["model_output"] or "").strip())
        is_homo.append(our_sentence in homo_set)

    print(f"Row-alignment check: {mismatches}/{len(mira)} mismatches after off-by-one correction.")
    if mismatches:
        print("  (non-zero here means the off-by-one assumption needs re-checking before trusting the rest)")

    # ── Core metrics (metrics.py) ───────────────────────────────────────────
    eval_results = stratified_evaluate(refs, hyps, is_homo)
    print_results(eval_results, title="Mira's Llama-3.1-8B zero-shot -- recomputed with our metrics.py")

    # ── PER (via our own G2P round-trip, same convention as zero-shot/run_baseline.py) ──
    from p2t_lora.data import g2p
    import jiwer

    def per(rs, hs):
        def to_ph(s):
            return " ".join(g2p.sentence_to_phoneme_list(s, stress=False))
        return jiwer.wer([to_ph(r) for r in rs], [to_ph(h) for h in hs]) * 100

    per_overall_val = per(refs, hyps)
    print(f"PER (overall, our G2P): {per_overall_val:.2f}%")

    # ── Extended metrics (SID / AER / WPER) ────────────────────────────────
    sid = ext.sid_breakdown(refs, hyps)
    aer = ext.allophonic_error_rate(refs, hyps)
    wper_h = ext.weighted_per(refs, hyps, method="heuristic") * 100
    try:
        wper_p = ext.weighted_per(refs, hyps, method="panphon") * 100
    except RuntimeError as e:
        wper_p = None
        print(f"WPER (panphon) skipped: {e}")

    print("\nSID:", sid)
    print("AER:", aer)
    print(f"WPER (heuristic): {wper_h:.2f}%")
    if wper_p is not None:
        print(f"WPER (panphon): {wper_p:.2f}%")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "source": "Mira Fleite, Llama-3.1-8B-Instruct, zero-shot prompting",
        "n": len(mira),
        "row_alignment_mismatches": mismatches,
        "core_metrics": eval_results,
        "PER_overall_pct": per_overall_val,
        "sid": sid,
        "aer": aer,
        "wper_heuristic_pct": wper_h,
        "wper_panphon_pct": wper_p,
    }
    out_path = os.path.join(OUT_DIR, "mira_recomputed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
