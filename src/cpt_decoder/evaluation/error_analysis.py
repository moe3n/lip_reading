"""
CPT Decoder — Error Pattern Analysis (Stage 2 + Stage 3-Option-2)
====================================================================
Implements the automatable slice of the "P2T Three-Stage Error Analysis
Framework" (P2T Error Pattern Analysis V1.pdf): Stage 2 (phoneme/word
error-pattern analysis) combined with Stage 3 Option 2 (dictionary-based
lexical/homophone auto-labelling).

WHY THIS SLICE, AND NOT THE WHOLE FRAMEWORK
--------------------------------------------
The framework's Stage 1 (PER/WER/CER) is already live in metrics.py.
Stage 3 Options 1/3/4/5 (manual annotation, grammar-based, semantic
similarity, LLM-based) either need new dependencies not installed in
this project (spaCy, sentence-transformers, etc.) or need human
judgement — they're deferred until the real Llama 3.2:3B run produces
errors worth that investment. The dry-run's errors (small stand-in
models, tens-to-hundreds of sentences) are dominated by undertrained-
model noise, per the earlier three-agent WER audit — not the kind of
systematic confusion this framework is built to detect.

This module is the part that's genuinely free right now: it reuses
jiwer (already a project dependency, already imported in metrics.py,
but only ever called for its scalar wer()/cer() — process_words() also
gives a full substitution/insertion/deletion breakdown with per-pair
word alignment) and hard_negatives.py's get_homophones() /
get_near_homophones() (already live, already used for training-time
hard-negative mining — repurposed here for evaluation-time
classification). Zero new dependencies.

WHAT IT'S FOR
-------------
The contrastive hard-negative mechanism in this architecture is
specifically designed to fix homophone-driven confusions. This module
answers the question that determines whether that's the right lever to
keep tuning: of the substitution errors a model makes, how many are
phonetically explainable (exact or near-homophone) vs. unrelated? If
errors cluster on homophones, that's a green light to keep tuning
contrastive hyperparameters (margin, negative count, mining density).
If errors are mostly unrelated substitutions, the bottleneck is more
likely model capacity / training data scale (i.e. needs the real GPU
run with more data and a far larger model), not more hard-negative
mining.
"""

import sys
import os
import re
from typing import List, Dict, Optional
from collections import Counter

import jiwer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cpt_decoder.augmentation.hard_negatives import get_homophones, get_near_homophones  # noqa: E402


# ── Text normalisation (mirrors metrics.py's normalise(), kept independent
#    on purpose so this module has no import-order dependency on metrics.py) ──
def normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Per-pair substitution classification ──────────────────────────────────────
def classify_substitution(ref_word: str, hyp_word: str) -> str:
    """
    Classify a single substitution (ref_word -> hyp_word) as one of:
        "Homophone"      : hyp_word is an exact CMU-dict homophone of ref_word
                            (identical phoneme sequence, stress stripped)
        "Near-homophone"  : hyp_word's phoneme sequence is within edit
                            distance 1 of ref_word's, but not identical
        "Other"           : no phonetic relationship found (or ref_word isn't
                            in the CMU dictionary at all, e.g. OOV / proper noun)

    get_homophones() is an O(1)-ish dict lookup (cheap); get_near_homophones()
    brute-force-scans the full ~125k-entry CMU dictionary, so it's only called
    when the cheap exact check has already failed — keeps this affordable to
    run over an entire validation set.
    """
    ref_word, hyp_word = ref_word.upper(), hyp_word.upper()
    if ref_word == hyp_word:
        return "Equal"  # shouldn't normally be reached from a 'substitute' chunk

    if hyp_word in get_homophones(ref_word):
        return "Homophone"

    near = get_near_homophones(ref_word, max_distance=1)
    near_words = {w for w, dist in near if dist >= 1}  # distance-0 entries already
                                                         # covered by get_homophones above
    if hyp_word in near_words:
        return "Near-homophone"

    return "Other"


# ── Per-sentence-pair error breakdown ─────────────────────────────────────────
def analyze_pair(reference: str, hypothesis: str) -> Dict:
    """
    Run jiwer's word-level alignment on a single (reference, hypothesis) pair
    and classify every substitution found.

    Returns:
        {
          "n_hits": int, "n_substitutions": int, "n_deletions": int, "n_insertions": int,
          "substitutions": [ {"ref": str, "hyp": str, "category": str}, ... ],
        }
    """
    ref_norm = normalise(reference)
    hyp_norm = normalise(hypothesis)
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    if not ref_words:
        return {"n_hits": 0, "n_substitutions": 0, "n_deletions": 0, "n_insertions": 0,
                "substitutions": []}

    out = jiwer.process_words([ref_norm], [hyp_norm])
    chunk = out.alignments[0]

    subs = []
    for c in chunk:
        if c.type != "substitute":
            continue
        ref_span = ref_words[c.ref_start_idx:c.ref_end_idx]
        hyp_span = hyp_words[c.hyp_start_idx:c.hyp_end_idx]
        for rw, hw in zip(ref_span, hyp_span):  # 1:1 within a substitute chunk
            subs.append({"ref": rw, "hyp": hw, "category": classify_substitution(rw, hw)})

    return {
        "n_hits":          out.hits,
        "n_substitutions": out.substitutions,
        "n_deletions":     out.deletions,
        "n_insertions":    out.insertions,
        "substitutions":   subs,
    }


# ── Full-set error category report ────────────────────────────────────────────
def error_category_report(all_refs: List[str],
                           all_hyps: List[str],
                           homo_mask: Optional[List[bool]] = None) -> Dict:
    """
    Run analyze_pair() across an entire evaluation set and aggregate.

    Args:
        all_refs  : reference sentences
        all_hyps  : decoded hypothesis sentences
        homo_mask : optional boolean list (True = sentence drawn from the
                    homophone-containing subset) — if given, the substitution
                    category breakdown is also reported split by this mask,
                    which is the comparison that actually matters for deciding
                    whether the contrastive hard-negative mechanism is doing
                    its job.

    Returns a dict with overall totals, a substitution-category breakdown,
    and (if homo_mask given) the same breakdown split by homo_mask.
    """
    def _accumulate(refs, hyps):
        totals = {"n_hits": 0, "n_substitutions": 0, "n_deletions": 0, "n_insertions": 0}
        cat_counts = Counter()
        examples = {"Homophone": [], "Near-homophone": [], "Other": []}
        for ref, hyp in zip(refs, hyps):
            res = analyze_pair(ref, hyp)
            for k in totals:
                totals[k] += res[k]
            for s in res["substitutions"]:
                cat_counts[s["category"]] += 1
                if s["category"] in examples and len(examples[s["category"]]) < 5:
                    examples[s["category"]].append((ref, hyp, s["ref"], s["hyp"]))
        return totals, cat_counts, examples

    overall_totals, overall_cats, overall_examples = _accumulate(all_refs, all_hyps)

    report = {
        "overall": {
            "totals": overall_totals,
            "substitution_categories": dict(overall_cats),
            "examples": overall_examples,
        }
    }

    if homo_mask is not None:
        homo_refs = [r for r, m in zip(all_refs, homo_mask) if m]
        homo_hyps = [h for h, m in zip(all_hyps, homo_mask) if m]
        non_refs  = [r for r, m in zip(all_refs, homo_mask) if not m]
        non_hyps  = [h for h, m in zip(all_hyps, homo_mask) if not m]

        h_totals, h_cats, h_ex = _accumulate(homo_refs, homo_hyps)
        n_totals, n_cats, n_ex = _accumulate(non_refs, non_hyps)
        report["homophone"]     = {"totals": h_totals, "substitution_categories": dict(h_cats), "examples": h_ex}
        report["non_homophone"] = {"totals": n_totals, "substitution_categories": dict(n_cats), "examples": n_ex}

    return report


# ── Pretty printer ────────────────────────────────────────────────────────────
def print_error_report(report: Dict, title: str = "Error Pattern Analysis") -> None:
    width = 66
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

    for key, label in [("overall", "Overall"), ("homophone", "Homophone subset"),
                        ("non_homophone", "Non-homophone subset")]:
        if key not in report:
            continue
        sect = report[key]
        t = sect["totals"]
        cats = sect["substitution_categories"]
        n_sub = t["n_substitutions"]
        print(f"\n  -- {label} --")
        print(f"     hits={t['n_hits']}  substitutions={t['n_substitutions']}  "
              f"deletions={t['n_deletions']}  insertions={t['n_insertions']}")
        if n_sub == 0:
            print("     (no substitutions to classify)")
            continue
        for cat in ["Homophone", "Near-homophone", "Other"]:
            n = cats.get(cat, 0)
            pct = (n / n_sub * 100) if n_sub else 0.0
            print(f"     {cat:<16} {n:>5} / {n_sub}  ({pct:5.1f}% of substitutions)")

    if "homophone" in report and report["overall"]["totals"]["n_substitutions"] > 0:
        h_cats = report["homophone"]["substitution_categories"]
        h_sub = report["homophone"]["totals"]["n_substitutions"]
        if h_sub > 0:
            phon_explained = h_cats.get("Homophone", 0) + h_cats.get("Near-homophone", 0)
            pct = phon_explained / h_sub * 100
            print(f"\n  -> Of substitution errors on homophone-containing sentences,")
            print(f"     {pct:.1f}% are phonetically explainable (exact or near-homophone).")
            print( "     High %: contrastive hard-negative tuning is the right lever.")
            print( "     Low %:  errors are likely scale/capacity-driven, not lexical —")
            print( "             the real Llama run is the more relevant fix, not more")
            print( "             hard-negative mining.")
    print("\n" + "=" * width + "\n")


if __name__ == "__main__":
    # ── Smoke test with toy data mirroring metrics.py's own smoke test ────────
    refs = [
        "I COULD LABEL THIS AS MEAT",
        "THEY FOUND THE SITE ON THE COAST",
        "WHAT REALLY MAKES A CHIP IS THE CRUNCH",
        "FRESH OUT THE FRYER",
        "THE TRADITIONAL CHIP PAN OFTEN STAYS ON THE SHELF",
    ]
    hyps = [
        "I COULD LABEL THIS AS MEET",          # exact homophone substitution
        "THEY FOUND THE SIGHT ON THE COAST",    # exact homophone substitution
        "WHAT REALLY MAKES A SHIP IS THE CRUNCH",  # near-homophone (CHIP -> SHIP, edit distance 1)
        "FRESH OUT THE DRYER",                  # also near-homophone (FRYER -> DRYER, edit distance 1, not unrelated)
        "THE TRADITIONAL BANANA PAN OFTEN STAYS ON THE SHELF",  # unrelated substitution (CHIP -> BANANA)
    ]
    homo_mask = [True, True, False, False, False]

    report = error_category_report(refs, hyps, homo_mask)
    print_error_report(report, "Smoke test — error_analysis.py")
