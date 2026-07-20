"""Stage 3 — extended metrics on dedup beam-5 predictions.

Implements Mira Fleite's methodology steps 4-6 on
`predictions_beam5.csv` (n=949, dedup val, beam-5), following the order
in `p2t_lora_checkpoints_dedup/NOTES.md`:

  Step 4 — WPER (weighted PER) using the project's own ARPAbet feature
           table, place=0.4/manner=0.4/voicing=0.2 (heuristic method,
           same as Mira's "Heuristic matrix").
  Step 5 — AER (allophonic error rate) by place/manner/voicing feature
           that differs, pooled over all phoneme substitutions on
           EM-False rows.
  Step 6 — Per-phoneme drill-down on the worst confusions, including
           the largest absolute counts and the largest within-phoneme
           error rate (errors / occurrences in the reference).

Plus a per-bucket error-type breakdown (vowel / manner / place / voicing)
stratified by homophone mask, so the failure modes from Step 1 (homophone
bucket leakage) can be cross-referenced with the articulatory dimension
that's breaking.

Reads predictions_beam5.csv, writes to analysis/tables/.
"""

import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
PREDICTIONS = ROOT / "predictions_beam5.csv"
OUT_DIR = ROOT / "analysis" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Run extended_metrics.py in the source tree. Imports follow the same
# sys.path dance the package itself uses.
SRC = ROOT.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from p2t_lora.evaluation.extended_metrics import (
    ARPABET_FEATURES,
    _dominant_feature,
    allophonic_error_rate,
    error_type_summary,
    sid_breakdown,
    weighted_per,
)
from p2t_lora.evaluation.metrics import normalise
from p2t_lora.augmentation.hard_negatives import get_homophones


def load_predictions() -> list[tuple[str, str, bool]]:
    """Returns [(target, prediction, is_homophone), ...] in CSV order."""
    rows = []
    with open(PREDICTIONS, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                (r["target"], r["prediction"], r["is_homophone"] == "True")
            )
    return rows


def split_em(rows):
    em, emf = [], []
    for tgt, hyp, homo in rows:
        if normalise(tgt) == normalise(hyp):
            em.append((tgt, hyp, homo))
        else:
            emf.append((tgt, hyp, homo))
    return em, emf


# ── Step 4: WPER ──────────────────────────────────────────────────────────────
def step4_wper():
    """Plain PER vs heuristic WPER, on overall / homophone / non-homophone
    subsets and on EM-False. The drop (or absence thereof) from plain to
    weighted PER is the headline: WPER << PER means errors are phonetically
    close (Mira's interpretation); WPER ≈ PER means errors are random
    substitutions. Numbers below 5% should be read as directional only —
    with only 949 val rows the absolute error count is small."""
    rows = load_predictions()
    em, emf = split_em(rows)

    # group: overall / homophone / non-homophone / EM-False / EM-True
    groups = {
        "overall": rows,
        "homophone": [r for r in rows if r[2]],
        "non_homophone": [r for r in rows if not r[2]],
        "em_false": emf,
        "em_true": em,
    }
    out = []
    for name, group in groups.items():
        if not group:
            continue
        refs = [r[0] for r in group]
        hyps = [r[1] for r in group]
        n = len(group)
        # plain PER (jiwer, char-level after phonemizing to phoneme strings)
        # Use jiwer.process_words on space-joined phonemes to get PER denominator.
        # Plain PER = (S+I+D)/N_ref phonemes.
        from p2t_lora.evaluation.extended_metrics import _phoneme_substitutions
        subs, n_ins, n_del, n_hits = _phoneme_substitutions(refs, hyps)
        n_ref_phones = n_hits + n_del + len(subs)
        per = (len(subs) + n_ins + n_del) / n_ref_phones if n_ref_phones else 0.0
        wper = weighted_per(refs, hyps, method="heuristic")
        # ratio WPER/PER — <1 means errors are phonetically close
        ratio = wper / per if per > 0 else 0.0
        out.append({
            "group": name, "n": n,
            "n_ref_phones": n_ref_phones,
            "n_subs": len(subs), "n_ins": n_ins, "n_del": n_del, "n_hits": n_hits,
            "per": per * 100,
            "wper_heuristic": wper * 100,
            "wper_per_ratio": ratio,
        })

    out_path = OUT_DIR / "wper_breakdown.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    return out


# ── Step 5: AER + dominant-feature per sentence ───────────────────────────────
def step5_aer():
    """AER by place/manner/voicing on EM-False rows (where phoneme
    substitutions actually exist), plus on each homophone mask subset
    of EM-False."""
    rows = load_predictions()
    _, emf = split_em(rows)
    homo_emf = [r for r in emf if r[2]]
    non_homo_emf = [r for r in emf if not r[2]]

    out = []
    for label, group in [
        ("em_false_overall", emf),
        ("em_false_homophone", homo_emf),
        ("em_false_non_homophone", non_homo_emf),
    ]:
        if not group:
            continue
        refs = [r[0] for r in group]
        hyps = [r[1] for r in group]
        r = allophonic_error_rate(refs, hyps)
        r["group"] = label
        r["n"] = len(group)
        out.append(r)

    out_path = OUT_DIR / "aer_breakdown.csv"
    fieldnames = [
        "group", "n", "n_substitutions", "n_classified",
        "n_insertions", "n_deletions", "n_hits",
        "place", "manner", "voicing",
        "place_pct", "manner_pct", "voicing_pct",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out)
    return out


# ── Step 6: per-phoneme drill-down ────────────────────────────────────────────
def step6_per_phoneme():
    """For each ARPAbet phoneme, count (1) absolute substitutions where it
    is the reference, (2) the most common hyp_phoneme it gets swapped for,
    and (3) the within-phoneme error rate. The phoneme error rate is the
    number of substitutions involving that phoneme divided by its total
    occurrences in the reference (counted via the same phonemization
    helper used elsewhere in this package).

    Output is sorted by absolute substitution count descending. Counts
    below 2 are flagged as low-confidence per `NOTES.md` rule #5."""
    rows = load_predictions()
    _, emf = split_em(rows)
    refs = [r[0] for r in emf]
    hyps = [r[1] for r in emf]

    # Collect reference phoneme occurrences (denominator for within-rate).
    from p2t_lora.evaluation.extended_metrics import _phonemize
    ref_phone_counts = Counter()
    for ref in refs:
        for p in _phonemize(ref):
            if p in ARPABET_FEATURES:
                ref_phone_counts[p] += 1

    # Collect (ref_phon -> hyp_phon) substitutions.
    from p2t_lora.evaluation.extended_metrics import _phoneme_substitutions
    subs, *_ = _phoneme_substitutions(refs, hyps)
    sub_counts: Counter = Counter()
    pair_counts: dict[str, Counter] = defaultdict(Counter)
    for rp, hp in subs:
        sub_counts[rp] += 1
        pair_counts[rp][hp] += 1

    # Build rows.
    out = []
    for phon, sub_n in sub_counts.most_common():
        ref_n = ref_phone_counts.get(phon, 0)
        top_hyp, top_n = pair_counts[phon].most_common(1)[0]
        out.append({
            "ref_phon": phon,
            "ref_occurrences": ref_n,
            "n_substitutions": sub_n,
            "within_phoneme_error_rate": sub_n / ref_n * 100 if ref_n else 0.0,
            "top_hyp_phon": top_hyp,
            "top_hyp_count": top_n,
            "low_confidence": "yes" if sub_n < 2 else "no",
        })

    out_path = OUT_DIR / "per_phoneme_drilldown.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    return out


# ── Error-type breakdown (bonus, fast variant) ───────────────────────────────
def _label_pair(ref: str, hyp: str) -> str:
    """Same precedence as extended_metrics.error_type_breakdown, but with the
    exact-homophone check only (no get_near_homophones brute force scan of
    the 125k-word CMU dict). The brute force was the bottleneck — it ran for
    every non-homophone substitution pair in error_type_breakdown, taking
    minutes on the full 949-row set. Here we keep the near-homophone
    dimension silent; the per-phoneme drilldown (Step 6) already shows
    whether substitutions are phonetically close."""
    import jiwer as _jiwer
    ref_n, hyp_n = normalise(ref), normalise(hyp)
    if ref_n == hyp_n:
        return "Exact match"
    word_out = _jiwer.process_words([ref_n], [hyp_n])
    if word_out.hits == 0:
        return "Hallucination"
    ref_words, hyp_words = ref_n.split(), hyp_n.split()
    homo_hit = any(
        hw.upper() in get_homophones(rw.upper())
        for c in word_out.alignments[0] if c.type == "substitute"
        for rw, hw in zip(ref_words[c.ref_start_idx:c.ref_end_idx],
                          hyp_words[c.hyp_start_idx:c.hyp_end_idx])
    )
    if homo_hit:
        return "Homophone"
    dominant = _dominant_feature(ref, hyp)
    if dominant == "vowel":
        return "Vowel"
    if dominant is not None:
        # manner / place / voicing — collapse to single bucket here, drill in AER
        return dominant.capitalize()
    return "Other"


def step_bonus_error_type():
    """Per-sentence error-type breakdown on overall / EM-False subsets.
    Cross-reference with the Stage 1 buckets: overlap between
    `homophone_substitution` rows and the dominant feature (vowel / manner /
    place / voicing) is the question."""
    rows = load_predictions()
    em, emf = split_em(rows)

    refs_all = [r[0] for r in rows]
    hyps_all = [r[1] for r in rows]
    labels_all = [_label_pair(r, h) for r, h in zip(refs_all, hyps_all)]

    refs_emf = [r[0] for r in emf]
    hyps_emf = [r[1] for r in emf]
    labels_emf = [_label_pair(r, h) for r, h in zip(refs_emf, hyps_emf)]

    summary_all = error_type_summary(labels_all)
    summary_emf = error_type_summary(labels_emf)

    out_path = OUT_DIR / "error_type_breakdown.csv"
    # union of labels across both subsets, ordered by overall count
    all_labels = list(summary_all.keys())
    for label in summary_emf:
        if label not in all_labels:
            all_labels.append(label)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "count_all", "pct_all", "count_em_false", "pct_em_false"])
        for label in all_labels:
            info = summary_all.get(label, {"count": 0, "pct": 0.0})
            emf_info = summary_emf.get(label, {"count": 0, "pct": 0.0})
            w.writerow([
                label, info["count"], f"{info['pct']:.2f}",
                emf_info["count"], f"{emf_info['pct']:.2f}",
            ])
    return summary_all, summary_emf


def main():
    print("== Stage 3: extended metrics on dedup beam-5 (n=949) ==")
    print()
    print("Step 4 — WPER vs plain PER:")
    wper_rows = step4_wper()
    for r in wper_rows:
        print(f"  {r['group']:<22s} n={r['n']:>4d}  PER={r['per']:.2f}%  "
              f"WPER={r['wper_heuristic']:.2f}%  ratio={r['wper_per_ratio']:.3f}")
    print(f"  -> wrote wper_breakdown.csv")
    print()

    print("Step 5 — AER (place/manner/voicing) on EM-False:")
    aer_rows = step5_aer()
    for r in aer_rows:
        print(f"  {r['group']:<28s} n={r['n']:>4d}  subs={r['n_substitutions']:>4d}  "
              f"place={r['place_pct']:.1f}%  manner={r['manner_pct']:.1f}%  "
              f"voicing={r['voicing_pct']:.1f}%")
    print(f"  -> wrote aer_breakdown.csv")
    print()

    print("Step 6 — per-phoneme drill-down (sorted by abs substitution count):")
    pp_rows = step6_per_phoneme()
    for r in pp_rows[:10]:
        print(f"  {r['ref_phon']:<3s} ref_n={r['ref_occurrences']:>4d}  subs={r['n_substitutions']:>3d}  "
              f"rate={r['within_phoneme_error_rate']:.2f}%  "
              f"top -> {r['top_hyp_phon']:<3s} (n={r['top_hyp_count']})"
              f"{'  [LOW-N]' if r['low_confidence'] == 'yes' else ''}")
    print(f"  -> wrote per_phoneme_drilldown.csv  ({len(pp_rows)} phonemes with subs)")
    print()

    print("Bonus — error-type breakdown:")
    try:
        s_all, s_emf = step_bonus_error_type()
        print(f"  overall (n={sum(v['count'] for v in s_all.values())}):")
        for label, info in s_all.items():
            print(f"    {label:<15s} {info['count']:>4d}  {info['pct']:>5.2f}%")
        print(f"  em_false (n={sum(v['count'] for v in s_emf.values())}):")
        for label, info in s_emf.items():
            print(f"    {label:<15s} {info['count']:>4d}  {info['pct']:>5.2f}%")
        print(f"  -> wrote error_type_breakdown.csv")
    except Exception as e:
        print(f"  skipped (exception): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
