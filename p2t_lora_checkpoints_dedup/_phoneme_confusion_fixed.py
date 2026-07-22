"""Phoneme confusion matrix, computed on phoneme TOKENS.

Replaces _phoneme_confusion.py, which joined each phoneme sequence into one
unbroken string before aligning and therefore compared individual letters
(producing pairs like A->E and H->R). This version splits on whitespace so
'AH N D' becomes three tokens and alignment happens between ARPAbet symbols.

Reads predictions_beam5_with_match.csv, which already carries transcribed
phonemes for both sides. No model or GPU needed.

Outputs to analysis/tables/:
    phoneme_confusion_tokens_all949.csv
    phoneme_confusion_tokens_emfalse.csv
"""

import csv
import re
from collections import Counter
from pathlib import Path

from jiwer import process_words

ROOT = Path(__file__).parent
PHON_CSV = ROOT / "predictions_beam5_with_match.csv"
TABLES = ROOT / "analysis" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def phoneme_tokens(phon_str):
    """'AH N D | F AO R' -> ['AH','N','D','F','AO','R'].

    Drops the word separator, strips stress digits, and removes the OOV
    marker '?' so an unresolvable word contributes no spurious token.
    """
    if not phon_str:
        return []
    toks = []
    for t in phon_str.replace("|", " ").split():
        t = re.sub(r"[012]", "", t).strip()
        if t and t != "?":
            toks.append(t)
    return toks


def substitution_pairs(ref_toks, hyp_toks):
    """One (ref, hyp) pair per aligned position inside substitute chunks.

    Insertions and deletions are skipped: they have no counterpart to pair
    with. Uneven substitute spans are matched left to right and the surplus
    on the longer side is dropped.
    """
    if not ref_toks:
        return []
    out = process_words([" ".join(ref_toks)], [" ".join(hyp_toks)])
    pairs = []
    for block in out.alignments:
        for chunk in block:
            if chunk.type != "substitute":
                continue
            r = ref_toks[chunk.ref_start_idx:chunk.ref_end_idx]
            h = hyp_toks[chunk.hyp_start_idx:chunk.hyp_end_idx]
            for i in range(min(len(r), len(h))):
                pairs.append((r[i], h[i]))
    return pairs


def count(population):
    pair_counts, ref_counts = Counter(), Counter()
    rows = rows_with_subs = 0
    with PHON_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            is_em = row.get("exact_match", "False").lower() in ("true", "1")
            if population == "emfalse" and is_em:
                continue
            rows += 1
            ref = phoneme_tokens(row["target_phonemes"])
            hyp = phoneme_tokens(row["prediction_phonemes"])
            ref_counts.update(ref)
            pairs = substitution_pairs(ref, hyp)
            if pairs:
                rows_with_subs += 1
                pair_counts.update(pairs)
    return pair_counts, ref_counts, rows, rows_with_subs


def write_csv(path, pair_counts, ref_counts):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ref_phon", "hyp_phon", "count", "ref_count", "pct_of_ref"])
        rows = [(rp, hp, n, ref_counts.get(rp, 0)) for (rp, hp), n in pair_counts.items()]
        rows.sort(key=lambda x: (-x[2], x[0], x[1]))
        for rp, hp, n, rc in rows:
            w.writerow([rp, hp, n, rc, f"{n / rc * 100:.2f}" if rc else ""])


def report(label, pair_counts, ref_counts, rows, rows_with_subs, k=10):
    total = sum(pair_counts.values())
    print(f"\n=== {label} ===")
    print(f"rows={rows}  rows_with_substitutions={rows_with_subs}  "
          f"total_substitutions={total}  unique_pairs={len(pair_counts)}  "
          f"distinct_ref_phonemes={len(ref_counts)}")
    print(f"\n{'rank':>4}  {'ref':>4} -> {'hyp':<4}  {'n':>3}  {'ref occurs':>10}  {'% of ref':>8}")
    for i, ((rp, hp), n) in enumerate(
            sorted(pair_counts.items(), key=lambda kv: (-kv[1], kv[0])), 1):
        if i > k:
            break
        rc = ref_counts.get(rp, 0)
        print(f"{i:>4}  {rp:>4} -> {hp:<4}  {n:>3}  {rc:>10}  "
              f"{(n / rc * 100 if rc else 0):>7.2f}%")


def main():
    for pop, label, fname in [
        ("all", "ALL 949 ROWS", "phoneme_confusion_tokens_all949.csv"),
        ("emfalse", "76 FAILING ROWS", "phoneme_confusion_tokens_emfalse.csv"),
    ]:
        pc, rc, rows, subs = count(pop)
        write_csv(TABLES / fname, pc, rc)
        report(label, pc, rc, rows, subs)
        print(f"\nwrote {TABLES / fname}")


if __name__ == "__main__":
    main()
