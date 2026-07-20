"""Step 3 (filter exploration): try three filter rules and see how
many real phoneme-recognition errors each one leaves behind.
"""
import csv
from collections import Counter

from jiwer import process_characters


def clean(s):
    out = []
    q = 0
    for ch in s or "":
        if ch in ("|", " ", "\t", "\n", "0", "1", "2"):
            continue
        if ch == "?":
            q += 1
            continue
        out.append(ch)
    return "".join(out), q


def norm_text(s):
    return " ".join((s or "").lower().split())


def wordset(s):
    return set(norm_text(s).split())


filters = {
    "all_em_false": lambda r: r["exact_match"].lower() in ("true", "1"),
    "text_ne_match": lambda r: norm_text(r["target"]) == norm_text(r["prediction"]),
    "wordset_match": lambda r: wordset(r["target"]) == wordset(r["prediction"]),
}

rows = list(csv.DictReader(open(
    r"p2t_lora_checkpoints_dedup\predictions_beam5_with_match.csv",
    encoding="utf-8", newline="",
)))

for name, drop in filters.items():
    pair_counts = Counter()
    ref_counts = Counter()
    rows_kept = 0
    rows_with_subs = 0
    q_total = 0
    for r in rows:
        if not r.get("target_phonemes"):
            continue
        if drop(r):
            continue
        rows_kept += 1
        ref, ref_q = clean(r["target_phonemes"])
        hyp, hyp_q = clean(r["prediction_phonemes"])
        q_total += ref_q + hyp_q
        ref_counts.update(ref)
        try:
            out = process_characters(ref, hyp)
        except Exception:
            continue
        pairs = []
        for blk in out.alignments:
            for ch in blk:
                if ch.type != "substitute":
                    continue
                rc0 = ref[ch.ref_start_idx:ch.ref_end_idx]
                hc0 = hyp[ch.hyp_start_idx:ch.hyp_end_idx]
                n = min(len(rc0), len(hc0))
                for i in range(n):
                    pairs.append((rc0[i], hc0[i]))
        if pairs:
            rows_with_subs += 1
            pair_counts.update(pairs)
    print(f"=== filter={name} ===")
    print(f"  rows kept            : {rows_kept}")
    print(f"  rows with subs       : {rows_with_subs}")
    print(f"  total subs           : {sum(pair_counts.values())}")
    print(f"  q_stripped           : {q_total}")
    print(f"  unique pairs         : {len(pair_counts)}")
    print(f"  unique refs          : {len(ref_counts)}")
    top = sorted(pair_counts.items(), key=lambda x: -x[1])[:10]
    print(f"  top-10 pairs         : {[(rp + chr(61) + hp, n) for (rp, hp), n in top]}")
    print()