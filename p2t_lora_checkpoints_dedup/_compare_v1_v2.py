"""Compare v1 and v2 classifier buckets against manual verdict on the
26-row audit set. A row 'agrees' if its auto_bucket matches the manually
identified bucket in audit_comment (i.e., v1 classified it the way a
human would).

This is the right cross-check: we're measuring whether the *rule*
produces a human-plausible bucket, not just whether it agrees with v1.
"""
import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
TABLES = ROOT / "analysis" / "tables"


def manual_bucket(audit_comment, default):
    """Extract the human-verified bucket from audit_comment."""
    c = audit_comment.lower()
    mapping = [
        (r"\btruncation\b", "truncation"),
        (r"\bboundary hallucination\b|\bboundary\b", "boundary_hallucination"),
        (r"\bnon-word spelling\b|\bnon word spelling\b|\bnon-word\b",
         "non_word_spelling"),
        (r"\bsuffix hallucination\b|\bsuffix\b", "suffix_hallucination"),
        (r"\bdigit/word rendering\b|\bdigit-word rendering\b|\bdigit.*word\b",
         "digit_word_rendering"),
        (r"\boov[- ]target.*substitution\b|\boov target\b",
         "oov_target_substitution"),
        (r"\bsemantic substitution\b", "semantic_substitution"),
        (r"\bhomophone substitution\b|\bnear[- ]homophone\b",
         "homophone_substitution"),
    ]
    for pat, b in mapping:
        if re.search(pat, c):
            return b
    return default


def load_v1():
    """v1 buckets live on failure_buckets.csv (per-row)."""
    with (TABLES / "failure_buckets.csv").open(encoding="utf-8", newline="") as f:
        return {r["target"].strip() + "\x00" + r["prediction"].strip(): r["bucket"]
                for r in csv.DictReader(f)}


def load_v2():
    with (TABLES / "failure_buckets_v2.csv").open(encoding="utf-8", newline="") as f:
        return {r["target"].strip() + "\x00" + r["prediction"].strip(): r["bucket"]
                for r in csv.DictReader(f)}


def main():
    v1 = load_v1()
    v2 = load_v2()
    audit_rows = list(csv.DictReader((TABLES / "top_failures.csv").open(encoding="utf-8", newline="")))
    audit_rows = [r for r in audit_rows if r["manual_agree"].strip().upper() == "DISAGREE"]

    agree_v1 = agree_v2 = 0
    n = len(audit_rows)
    flips = Counter()
    for r in audit_rows:
        k = r["target"].strip() + "\x00" + r["prediction"].strip()
        truth = manual_bucket(r["audit_comment"], r["auto_bucket"])
        b1 = v1.get(k, "MISSING")
        b2 = v2.get(k, "MISSING")
        if b1 == truth:
            agree_v1 += 1
        if b2 == truth:
            agree_v2 += 1
        flips[(b1, b2)] += 1

    print(f"Audited DISAGREE rows: {n}")
    print(f"v1 agree with manual truth: {agree_v1}/{n} = {100.0*agree_v1/n:.1f}%")
    print(f"v2 agree with manual truth: {agree_v2}/{n} = {100.0*agree_v2/n:.1f}%")
    print(f"v2 - v1 delta            : {agree_v2 - agree_v1:+d} rows")
    print()
    print("v1 -> v2 flip pattern on DISAGREE rows:")
    for (b1, b2), c in sorted(flips.items(), key=lambda kv: -kv[1]):
        if b1 != b2:
            print(f"  {b1:<28} -> {b2:<28}  {c}")


if __name__ == "__main__":
    main()