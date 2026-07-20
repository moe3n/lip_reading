"""Classify failing (EM=False) rows into failure-mode buckets.

Reads predictions_beam5_with_match.csv (already includes target_phonemes and
prediction_phonemes with CMU OOV marker '?'). Writes:
  - analysis/tables/failure_buckets.csv  (per-row bucket assignment)
  - analysis/tables/bucket_counts.csv    (count per bucket, EM-False only)
"""

import csv
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent
SRC = ROOT / "predictions_beam5_with_match.csv"
OUT_DIR = ROOT / "analysis" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PER_ROW = OUT_DIR / "failure_buckets.csv"
OUT_COUNTS = OUT_DIR / "bucket_counts.csv"

DIGIT_RE = re.compile(r"\d")
WORD_RE = re.compile(r"[A-Za-z']+")


def phoneme_words(phon_str: str) -> list[str]:
    """phoneme column format: 'W1 | W2 | W3'. Return ['W1','W2','W3']."""
    return [w.strip() for w in phon_str.split("|") if w.strip()]


def has_oov(phon_str: str) -> bool:
    return "?" in phon_str.split()  # only '?' as standalone token


def text_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def classify(row: dict) -> tuple[str, str]:
    """Return (bucket, reason). Buckets:
       boundary_hallucination, non_word_spelling, truncation,
       suffix_hallucination, digit_word_rendering, homophone_substitution,
       semantic_substitution, oov_target_substitution, unclassified.
    """
    tgt = row["target"].strip()
    hyp = row["prediction"].strip()
    tgt_ph = row["target_phonemes"]
    hyp_ph = row["prediction_phonemes"]
    tgt_words = text_words(tgt)
    hyp_words = text_words(hyp)
    tgt_phs = phoneme_words(tgt_ph)
    hyp_phs = phoneme_words(hyp_ph)

    tgt_oov = has_oov(tgt_ph)
    hyp_oov = has_oov(hyp_ph)
    tgt_has_digit = bool(DIGIT_RE.search(tgt))
    hyp_has_digit = bool(DIGIT_RE.search(hyp))

    # --- digit/word rendering ------------------------------------------------
    if tgt_has_digit != hyp_has_digit:
        # one side is digit, other is word(s) — and length differs by 1 word
        if abs(len(tgt_words) - len(hyp_words)) <= 1:
            return ("digit_word_rendering",
                    f"digit/word side mismatch (tgt_digit={tgt_has_digit}, hyp_digit={hyp_has_digit})")

    # --- non-word spelling (prediction side OOV) -----------------------------
    if hyp_oov and not tgt_oov:
        # If the target has multiple words and the prediction differs a lot,
        # prefer 'semantic_substitution'; otherwise it's a spelling error.
        # Use heuristic: if target and prediction have same number of phoneme
        # words, it's spelling; else semantic.
        if len(tgt_phs) == len(hyp_phs):
            return ("non_word_spelling",
                    f"prediction has CMU OOV; phoneme-word counts equal ({len(tgt_phs)})")
        else:
            return ("semantic_substitution",
                    f"prediction has CMU OOV AND phoneme-word count differs (tgt={len(tgt_phs)} hyp={len(hyp_phs)})")

    # --- truncation ----------------------------------------------------------
    if len(hyp_words) < len(tgt_words):
        # If the first len(hyp_words) of target are mostly equal to hyp
        prefix_match = sum(
            1 for a, b in zip(tgt_words[:len(hyp_words)], hyp_words)
            if a.upper() == b.upper()
        )
        if prefix_match >= max(1, len(hyp_words) - 1):
            return ("truncation",
                    f"prediction shorter ({len(hyp_words)} vs {len(tgt_words)}), shared prefix {prefix_match}/{len(hyp_words)}")

    # --- suffix hallucination (-ING, -S, -ED, -LY, -MENT, -TION) -------------
    SUFFIXES = ("ING", "S", "ED", "LY", "MENT", "TION", "NESS", "ITY", "OUS")
    for suf in SUFFIXES:
        if (hyp.upper().endswith(suf) and
                len(hyp_words) >= len(tgt_words) and
                not tgt.upper().endswith(suf)):
            # The last prediction word is target word + suffix
            last_hyp = hyp_words[-1].upper()
            for tw in tgt_words[::-1]:
                if last_hyp.startswith(tw.upper()) and len(last_hyp) > len(tw):
                    return ("suffix_hallucination",
                            f"prediction word '{hyp_words[-1]}' = target '{tw}' + '{suf}'")
            break

    # --- boundary hallucination (space split/merge) --------------------------
    # If target and prediction share the same letter string but with a space
    # inserted/removed.
    tgt_no_space = tgt.replace(" ", "").replace("'", "").upper()
    hyp_no_space = hyp.replace(" ", "").replace("'", "").upper()
    if tgt_no_space == hyp_no_space and tgt != hyp:
        return ("boundary_hallucination",
                f"same letters, space differs (tgt_words={len(tgt_words)} hyp_words={len(hyp_words)})")

    # --- OOV target substitution ---------------------------------------------
    if tgt_oov and not hyp_oov:
        return ("oov_target_substitution",
                f"target has CMU OOV, prediction is real word")

    # --- semantic vs homophone substitution ----------------------------------
    # Both real words. Use phoneme overlap.
    if tgt_phs and hyp_phs:
        overlap = len(set(tgt_phs) & set(hyp_phs))
        total = max(len(set(tgt_phs)), len(set(hyp_phs)))
        ratio = overlap / total if total else 0.0
        # High overlap but small phoneme diff -> homophone/near-homophone
        # Low overlap -> semantic substitution
        if ratio >= 0.5:
            return ("homophone_substitution",
                    f"phoneme overlap {overlap}/{total}={ratio:.2f} (high)")
        else:
            return ("semantic_substitution",
                    f"phoneme overlap {overlap}/{total}={ratio:.2f} (low)")

    return ("unclassified", "no rule matched")


def main():
    with SRC.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # Classify all rows; we'll only report/keep EM-False ones for the count
    # table, but keep the per-row output with all rows so future passes can
    # pivot.
    annotated = []
    for r in rows:
        bucket, reason = classify(r)
        new = dict(r)
        new["bucket"] = bucket
        new["bucket_reason"] = reason
        annotated.append(new)

    # Per-row output
    fieldnames = list(annotated[0].keys())
    with OUT_PER_ROW.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(annotated)
    print(f"wrote {OUT_PER_ROW.name}  rows={len(annotated)}")

    # Count table restricted to EM-False
    em_false = [r for r in annotated if r["exact_match"] == "False"]
    counts = Counter(r["bucket"] for r in em_false)
    total_em_false = len(em_false)
    print(f"\nEM-False rows: {total_em_false}")
    print(f"{'bucket':<30}  n    pct")
    print("-" * 50)
    for b, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * n / total_em_false if total_em_false else 0.0
        print(f"{b:<30}  {n:>3}  {pct:>5.1f}%")
    unclass = counts.get("unclassified", 0)
    if unclass:
        print(f"\nunclassified rows (need manual review): {unclass}")
        print("first 5 unclassified (target | prediction):")
        for r in [x for x in em_false if x["bucket"] == "unclassified"][:5]:
            print(f"  T: {r['target']}")
            print(f"  P: {r['prediction']}")
            print()

    # Persist count table
    with OUT_COUNTS.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "n", "pct_of_em_false"])
        for b, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * n / total_em_false if total_em_false else 0.0
            w.writerow([b, n, f"{pct:.2f}"])
        w.writerow(["TOTAL", total_em_false, "100.00"])
    print(f"\nwrote {OUT_COUNTS.name}")


if __name__ == "__main__":
    main()