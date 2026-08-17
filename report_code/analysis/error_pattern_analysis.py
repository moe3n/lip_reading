"""Reusable three-stage error pattern analysis."""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from p2t_lora.evaluation.metrics import stratified_evaluate, normalise
from p2t_lora.evaluation import extended_metrics as em
from p2t_lora.evaluation.error_analysis import error_category_report

VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
          "IH", "IY", "OW", "OY", "UH", "UW"}

def load_predictions(path):
    """Return (refs, hyps, homo_mask, exact_flags). Tolerates the three column"""
    refs, hyps, homo, exact = [], [], [], []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "target" not in reader.fieldnames or "prediction" not in reader.fieldnames:
            sys.exit(f"predictions file must have 'target' and 'prediction' columns; "
                     f"found {reader.fieldnames}")
        for r in reader:
            t = (r.get("target") or "").strip()
            p = (r.get("prediction") or "").strip()
            refs.append(t)
            hyps.append(p)
            hv = str(r.get("is_homophone", "")).strip().lower()
            homo.append(hv in ("true", "1", "yes"))
            if "exact_match" in r and r["exact_match"] != "":
                exact.append(str(r["exact_match"]).strip().lower() in ("true", "1"))
            else:
                exact.append(normalise(t) == normalise(p))
    return refs, hyps, homo, exact

def stage1(refs, hyps, homo):
    """Conventional evaluation: WER/CER/exact match, plus PER from the pooled"""
    res = stratified_evaluate(refs, hyps, homo)
    subs, n_ins, n_del, n_hits = em._phoneme_substitutions(refs, hyps)
    n_ref = len(subs) + n_del + n_hits
    per = (len(subs) + n_ins + n_del) / n_ref * 100 if n_ref else 0.0
    overall = res["overall"]
    return {
        "phoneme_error_rate_pct": round(per, 3),
        "word_error_rate_pct": round(overall["WER"] * 100, 3),
        "char_error_rate_pct": round(overall["CER"] * 100, 3),
        "exact_match_pct": round(overall["Exact_Match"] * 100, 3),
        "n": len(refs),
    }

def stage2(refs, hyps, homo, out_dir):
    """Stage 2: phoneme error patterns (SID counts, confusion matrix, per-phoneme rates, articulatory features)."""
    sid = em.sid_breakdown(refs, hyps)

    subs, n_ins, n_del, n_hits = em._phoneme_substitutions(refs, hyps)
    pair_counts = Counter(subs)
    ref_counts = Counter(rp for rp, _ in subs)

    with open(os.path.join(out_dir, "stage2_confusion_pairs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ref_phoneme", "predicted_phoneme", "count"])
        for (rp, hp), n in sorted(pair_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            w.writerow([rp, hp, n])

    ref_total = Counter()
    for r in refs:
        ref_total.update(em._phonemize(r))
    per_phoneme = []
    for ph, sub_n in ref_counts.most_common():
        occ = ref_total.get(ph, sub_n)
        per_phoneme.append((ph, occ, sub_n, round(sub_n / occ * 100, 2) if occ else 0.0))
    per_phoneme.sort(key=lambda x: -x[3])
    with open(os.path.join(out_dir, "stage2_per_phoneme.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phoneme", "occurrences", "substitutions", "rate_pct"])
        w.writerows(per_phoneme)

    aer = em.allophonic_error_rate(refs, hyps)
    per = stage1(refs, hyps, homo)["phoneme_error_rate_pct"]
    wper = em.weighted_per(refs, hyps, method="heuristic")

    draw_confusion_matrix(pair_counts, os.path.join(out_dir, "stage2_confusion_matrix.png"))

    n_vv = sum(n for (rp, hp), n in pair_counts.items() if rp in VOWELS and hp in VOWELS)
    return {
        "sid": sid,
        "n_substitutions": len(subs),
        "n_distinct_pairs": len(pair_counts),
        "n_repeated_pairs": sum(1 for v in pair_counts.values() if v > 1),
        "vowel_to_vowel_subs": n_vv,
        "aer": {k: aer[k] for k in ("place_pct", "manner_pct", "voicing_pct", "n_classified")},
        "phoneme_error_rate_pct": per,
        "weighted_per_pct": round(wper * 100, 3),
        "wper_over_per": round((wper * 100) / per, 3) if per else 0.0,
    }

def stage3(refs, hyps, homo, exact, out_dir, do_semantic):
    """Stage 3: hierarchical error analysis (lexical, contextual, and semantic classification)."""
    report = error_category_report(refs, hyps, homo)
    cats = report["overall"]["substitution_categories"]
    contextual = report["overall"].get("stage3_categories", {})
    contextual_methods = report["overall"].get("stage3_methods", {})

    import re as _re
    up = sum(1 for h in hyps if any(w.isupper() and w.isalpha() and len(w) > 1 for w in h.split()))
    dig = sum(1 for h in hyps if _re.search(r"\d", h))

    labels = em.error_type_breakdown(refs, hyps)
    type_summary = em.error_type_summary(labels)

    fail_path = os.path.join(out_dir, "failing_rows.csv")
    n_fail = 0
    with open(fail_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target", "prediction", "is_homophone"])
        for r, h, hm, ex in zip(refs, hyps, homo, exact):
            if not ex:
                n_fail += 1
                w.writerow([r, h, hm])

    out = {
        "lexical_substitution_categories": cats,
        "contextual_resolutions": dict(contextual),
        "contextual_methods": dict(contextual_methods),
        "surface_audit": {"predictions_with_uppercase_word": up,
                          "predictions_with_digit": dig, "n_rows": len(hyps)},
        "phoneme_error_types": type_summary,
        "n_failing_rows": n_fail,
        "manual_audit": "pending (see failing_rows.csv; the framework's most "
                        "defensible option requires a human pass)",
    }
    if do_semantic:
        fail_refs = [r for r, ex in zip(refs, exact) if not ex]
        fail_hyps = [h for h, ex in zip(hyps, exact) if not ex]
        if fail_refs:
            out["semantic_bertscore_f1_failing"] = round(em.semantic_similarity(fail_refs, fail_hyps), 4)
    return out

def draw_confusion_matrix(pair_counts, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not pair_counts:
        return
    refs = {rp for rp, _ in pair_counts}
    hyps = {hp for _, hp in pair_counts}
    order = lambda s: sorted(x for x in s if x in VOWELS) + sorted(x for x in s if x not in VOWELS)
    row_labels, col_labels = order(refs), order(hyps)
    nrv = sum(1 for x in row_labels if x in VOWELS)
    ncv = sum(1 for x in col_labels if x in VOWELS)
    ri = {s: i for i, s in enumerate(row_labels)}
    ci = {s: i for i, s in enumerate(col_labels)}

    M = np.zeros((len(row_labels), len(col_labels)), dtype=int)
    for (rp, hp), n in pair_counts.items():
        M[ri[rp], ci[hp]] = n

    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 0.4), max(5, len(row_labels) * 0.4)))
    masked = np.ma.masked_where(M == 0, M)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#f5f5f5")
    ax.imshow(masked, cmap=cmap, vmin=1, vmax=max(2, M.max()), aspect="equal")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=90)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Predicted phoneme")
    ax.set_ylabel("Reference phoneme")
    ax.set_title("Phoneme substitution confusion matrix\nvowels grouped top-left")
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if M[i, j] > 0:
                ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=8,
                        color="black" if M[i, j] < 2 else "white")
    ax.axhline(nrv - 0.5, color="#c0392b", lw=1.2)
    ax.axvline(ncv - 0.5, color="#c0392b", lw=1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Three-stage P2T error pattern analysis")
    ap.add_argument("--predictions", required=True, help="predictions CSV (target, prediction columns)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--stages", default="1,2,3",
                    help="which stages to run this pass, e.g. '1' or '1,2,3'. "
                         "Runs incrementally: results merge into an existing summary.json "
                         "in --out, so stages can be done one at a time.")
    ap.add_argument("--semantic", action="store_true",
                    help="Stage 3 only: also run BERTScore on failing rows (downloads a model, slower)")
    args = ap.parse_args()
    stages = {s.strip() for s in args.stages.split(",")}

    Path(args.out).mkdir(parents=True, exist_ok=True)
    refs, hyps, homo, exact = load_predictions(args.predictions)
    print(f"Loaded {len(refs)} rows | {sum(exact)} exact match | {len(refs) - sum(exact)} failing")

    summary_path = os.path.join(args.out, "summary.json")
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    summary["source"] = os.path.abspath(args.predictions)
    summary["n_rows"] = len(refs)

    if "1" in stages:
        summary["stage1_conventional"] = stage1(refs, hyps, homo)
        s1 = summary["stage1_conventional"]
        print(f"\nStage 1  PER {s1['phoneme_error_rate_pct']}%  WER {s1['word_error_rate_pct']}%  "
              f"CER {s1['char_error_rate_pct']}%  EM {s1['exact_match_pct']}%")
    if "2" in stages:
        summary["stage2_phoneme_patterns"] = stage2(refs, hyps, homo, args.out)
        s2 = summary["stage2_phoneme_patterns"]
        print(f"Stage 2  {s2['n_substitutions']} phoneme subs across {s2['n_distinct_pairs']} pairs, "
              f"{s2['n_repeated_pairs']} repeated; WPER/PER {s2['wper_over_per']}")
    if "3" in stages:
        summary["stage3_hierarchical"] = stage3(refs, hyps, homo, exact, args.out, args.semantic)
        print(f"Stage 3  {summary['stage3_hierarchical']['n_failing_rows']} failing rows "
              f"(manual audit pending)")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nStages {sorted(stages)} written to {args.out}/")

if __name__ == "__main__":
    main()
