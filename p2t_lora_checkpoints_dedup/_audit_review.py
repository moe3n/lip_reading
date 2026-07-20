"""Stage 1: pull the audit-corrected failure-mode counts from top_failures.csv.
Goal: re-tally the 26 audited rows using the manual bucket mentioned in
audit_comment, so we know what the *true* distribution looks like.
"""
import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "analysis" / "tables" / "top_failures.csv"
OUT = ROOT / "analysis" / "tables" / "bucket_counts_audited.csv"


def manual_bucket(audit_comment, auto_bucket):
    """If audit agrees, return auto_bucket. If disagrees, try to
    extract the manually-suggested bucket from the audit_comment.

    Audit comments use phrases like:
      "truncation: model dropped FORT- syllable..."
      "boundary hallucination: ..."
      "non-word spelling: ..."
      "semantic substitution: ..."
      "homophone substitution: ..."
      "OOV target -> plausible-English substitution: ..."
      "digit/word rendering: ..."
      "suffix hallucination: ..."
      "vowel-drop non-word: ..."
      "plural hallucination: ..."
      "consonant-cluster substitution: ..."
      "consonant-drop near-homophone: ..."
      "syllable collapse: ..."
    Map each to one of the 8 canonical buckets.
    """
    c = audit_comment.lower()
    # Canonical phrases that map directly
    mapping = [
        (r"\btruncation\b", "truncation"),
        (r"\bboundary hallucination\b", "boundary_hallucination"),
        (r"\bboundary\b", "boundary_hallucination"),
        (r"\bnon-word spelling\b|\bnon word spelling\b|\bnon-word\b", "non_word_spelling"),
        (r"\bsuffix hallucination\b|\bsuffix\b", "suffix_hallucination"),
        (r"\bdigit/word rendering\b|\bdigit-word rendering\b|\bdigit.*word\b", "digit_word_rendering"),
        (r"\boov[- ]target.*substitution\b|\boov target\b", "oov_target_substitution"),
        (r"\bsemantic substitution\b", "semantic_substitution"),
        (r"\bhomophone substitution\b|\bnear[- ]homophone\b", "homophone_substitution"),
    ]
    for pat, bucket in mapping:
        if re.search(pat, c):
            return bucket
    # Default: keep auto
    return auto_bucket


def main():
    rows = list(csv.DictReader(SRC.open(encoding="utf-8", newline="")))
    n_agree = sum(1 for r in rows if r["manual_agree"].strip().upper() == "AGREE")
    n_dis = sum(1 for r in rows if r["manual_agree"].strip().upper() == "DISAGREE")
    print(f"audited rows: {len(rows)}  AGREE={n_agree}  DISAGREE={n_dis}")
    print()

    auto_dist = Counter(r["auto_bucket"] for r in rows)
    manual_dist = Counter(manual_bucket(r["audit_comment"], r["auto_bucket"]) for r in rows)
    total = len(rows)

    print(f"{'bucket':<28} {'auto':>6} {'auto%':>7} {'manual':>7} {'manual%':>8}  delta")
    print("-" * 70)
    for b in sorted(set(auto_dist) | set(manual_dist), key=lambda x: -manual_dist.get(x, 0)):
        a = auto_dist.get(b, 0)
        m = manual_dist.get(b, 0)
        ap = 100.0 * a / total
        mp = 100.0 * m / total
        print(f"{b:<28} {a:>6} {ap:>6.1f}% {m:>7} {mp:>7.1f}%  {m - a:+d}")
    print("-" * 70)
    print(f"{'TOTAL':<28} {sum(auto_dist.values()):>6} {sum(manual_dist.values()):>7}")

    # Write the audited tally to CSV
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "auto_n", "auto_pct", "manual_n", "manual_pct"])
        for b in sorted(set(auto_dist) | set(manual_dist), key=lambda x: -manual_dist.get(x, 0)):
            a = auto_dist.get(b, 0)
            m = manual_dist.get(b, 0)
            w.writerow([b, a, f"{100.0*a/total:.2f}", m, f"{100.0*m/total:.2f}"])
        w.writerow(["TOTAL", total, "100.00", total, "100.00"])
    print()
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()