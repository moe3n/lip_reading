"""Step 2: count substitution events across all 949 dedup val rows
and across the 76 EM-False subset, then dump two long-form CSVs to
analysis/tables/.

Output files (long-form: one row per ref_phon/hyp_phon pair):
  analysis/tables/phoneme_confusion_all949.csv
  analysis/tables/phoneme_confusion_emfalse.csv

Selection criterion for top-10 heatmap axes: substitution mass
(total count of times the ref phoneme was swapped, regardless of
target hyp). Ties are broken by reference counts (how often the
phoneme appears in target streams across the population).

After this step, the script prints the top-20 substitution pairs
per population so the user can sign off before heatmap rendering.
"""

import csv
from collections import Counter
from pathlib import Path

from jiwer import process_characters

ROOT = Path(__file__).parent
PHON_CSV = ROOT / "predictions_beam5_with_match.csv"
TABLES = ROOT / "analysis" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def clean_phoneme_stream(phon_str):
    """Strip word-sep '|', whitespace, stress digits 0/1/2, and OOV
    marker '?'. Returns (stream, q_count).
    """
    out = []
    q_count = 0
    for ch in phon_str or "":
        if ch in ("|", " ", "\t", "\n", "0", "1", "2"):
            continue
        if ch == "?":
            q_count += 1
            continue
        out.append(ch)
    return "".join(out), q_count


def substitution_pairs(ref_stream, hyp_stream):
    """One (ref_phon, hyp_phon) pair per aligned position inside
    substitute chunks. Inserts and deletes are skipped. Unequal
    sub spans are paired left-aligned; unmatched trailing chars on
    the longer side are silently dropped.
    """
    out_alignment = process_characters(ref_stream, hyp_stream)
    pairs = []
    for block in out_alignment.alignments:
        for chunk in block:
            if chunk.type != "substitute":
                continue
            ref_chars = ref_stream[chunk.ref_start_idx:chunk.ref_end_idx]
            hyp_chars = hyp_stream[chunk.hyp_start_idx:chunk.hyp_end_idx]
            n = min(len(ref_chars), len(hyp_chars))
            for i in range(n):
                pairs.append((ref_chars[i], hyp_chars[i]))
    return pairs


def process_row(target_phonemes, prediction_phonemes):
    ref_stream, ref_q = clean_phoneme_stream(target_phonemes)
    hyp_stream, hyp_q = clean_phoneme_stream(prediction_phonemes)
    pairs = substitution_pairs(ref_stream, hyp_stream)
    return pairs, ref_stream, hyp_stream, ref_q, hyp_q


def iter_rows():
    """Yield each row dict with is_em flag and parsed phonemes."""
    with PHON_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            r["is_em"] = r.get("exact_match", "False").lower() in ("true", "1")
            yield r


def count_substitutions(population_filter):
    """population_filter(row) -> bool. Returns:
       pair_counts: Counter[(ref_phon, hyp_phon) -> n]
       ref_counts : Counter[ref_phon -> n_occurrences_in_target_stream]
       rows_seen  : int
       rows_with_subs : int
       q_total     : int  (sum of ref_q + hyp_q across rows)
    """
    pair_counts = Counter()
    ref_counts = Counter()
    rows_seen = 0
    rows_with_subs = 0
    q_total = 0
    for r in iter_rows():
        if not population_filter(r):
            continue
        rows_seen += 1
        pairs, ref_s, _, ref_q, hyp_q = process_row(
            r["target_phonemes"], r["prediction_phonemes"]
        )
        q_total += ref_q + hyp_q
        ref_counts.update(ref_s)
        if pairs:
            rows_with_subs += 1
            pair_counts.update(pairs)
    return pair_counts, ref_counts, rows_seen, rows_with_subs, q_total


def write_long_form_csv(path, pair_counts, ref_counts):
    """One row per (ref_phon, hyp_phon) pair with n and ref_count."""
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ref_phon", "hyp_phon", "count", "ref_count"])
        # sort: count desc, then ref_phon asc, hyp_phon asc
        rows = []
        for (rp, hp), n in pair_counts.items():
            rows.append((rp, hp, n, ref_counts.get(rp, 0)))
        rows.sort(key=lambda x: (-x[2], x[0], x[1]))
        for rp, hp, n, rc in rows:
            w.writerow([rp, hp, n, rc])


def print_top_pairs(label, pair_counts, ref_counts, k=20):
    print(f"=== {label} ===")
    print(f"{'rank':>4}  {'ref':>3}  {'hyp':>3}  {'count':>6}  {'ref_count':>10}")
    items = sorted(
        pair_counts.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
    )
    for i, ((rp, hp), n) in enumerate(items[:k], 1):
        print(f"{i:>4}  {rp:>3}  {hp:>3}  {n:>6}  {ref_counts.get(rp, 0):>10}")
    print(f"... total unique (ref,hyp) pairs: {len(pair_counts)}")
    print()


def main():
    # Population 1: all 949 dedup val rows
    pc_all, rc_all, seen_all, sub_all, q_all = count_substitutions(
        lambda r: True
    )
    write_long_form_csv(
        TABLES / "phoneme_confusion_all949.csv", pc_all, rc_all
    )
    print(
        f"[all949] rows={seen_all}  rows_with_subs={sub_all}  "
        f"total_subs={sum(pc_all.values())}  q_stripped={q_all}  "
        f"unique_pairs={len(pc_all)}  unique_refs={len(rc_all)}"
    )
    print_top_pairs("ALL 949 — top-20 substitution pairs", pc_all, rc_all, k=20)

    # Population 2: the 76 EM-False rows
    pc_emf, rc_emf, seen_emf, sub_emf, q_emf = count_substitutions(
        lambda r: not r["is_em"]
    )
    write_long_form_csv(
        TABLES / "phoneme_confusion_emfalse.csv", pc_emf, rc_emf
    )
    print(
        f"[emfalse] rows={seen_emf}  rows_with_subs={sub_emf}  "
        f"total_subs={sum(pc_emf.values())}  q_stripped={q_emf}  "
        f"unique_pairs={len(pc_emf)}  unique_refs={len(rc_emf)}"
    )
    print_top_pairs("EM-FALSE 76 — top-20 substitution pairs", pc_emf, rc_emf, k=20)


if __name__ == "__main__":
    main()
