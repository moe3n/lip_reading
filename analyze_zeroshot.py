"""
Offline analyzer for zero-shot baseline predictions.
====================================================
Reads the jsonl files produced by zero-shot/run_baseline.py and computes
metrics + Stage 2/3 error analysis + extended (SID/AER/WPER) without
touching the GPU. Multi-process: the dominant cost is the per-substitution
near-homophone scan (Stage 2 ~ 125k-word CMU dict, ~1s each on CPU),
which we parallelise across N workers via multiprocessing.Pool.

Output filenames mirror what run_baseline.py would have written inline:
    errors_<split>_<n>_<mode>.json
    extended_<split>_<n>_<mode>.json
    metrics_<split>_<n>.csv        (appends to / creates)
    view_<split>_<n>_<mode>.txt

Usage:
    .venv\\Scripts\\python.exe analyze_zeroshot.py ^
        --jsonl zero-shot\\baseline\\preds_train_45839_clean.jsonl ^
        --mode clean ^
        --workers 8
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool
from typing import List, Tuple

# Same sys.path manipulation run_baseline.py uses
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from p2t_lora.data import loader as data_loader        # noqa: E402
from p2t_lora.data import g2p                           # noqa: E402
from p2t_lora.evaluation.metrics import (               # noqa: E402
    word_error_rate, character_error_rate, bleu4_score, exact_match,
)
from p2t_lora.evaluation import extended_metrics as ext  # noqa: E402
from p2t_lora.evaluation.error_analysis import (        # noqa: E402
    analyze_pair,
)

TRAIN_N, VAL_N, TEST_N = 45839, 1082, 1243


# ── Path helpers ─────────────────────────────────────────────────────────────
def parse_filename(jsonl_path: str) -> Tuple[str, str, int, str]:
    """Extract (split, mode, n, view_stem) from preds_<split>_<n>_<mode>.jsonl."""
    base = os.path.basename(jsonl_path)
    m = re.match(r"preds_(train|val|test)_(\d+)_(clean|raw)\.jsonl$", base)
    if not m:
        raise ValueError(f"unexpected jsonl name: {base}")
    split, n, mode = m.group(1), int(m.group(2)), m.group(3)
    return split, mode, n, os.path.dirname(jsonl_path)


# ── Core metrics (cheap, single-process) ─────────────────────────────────────
def phoneme_error_rate(refs, hyps) -> float:
    def to_ph(s):
        return " ".join(g2p.sentence_to_phoneme_list(s, stress=False))
    import jiwer
    return jiwer.wer([to_ph(r) for r in refs], [to_ph(h) for h in hyps])


def stratify(refs, hyps, is_homo):
    homo = [(r, h) for r, h, m in zip(refs, hyps, is_homo) if m]
    non_homo = [(r, h) for r, h, m in zip(refs, hyps, is_homo) if not m]
    return {
        "Overall": (refs, hyps),
        "Homophone": tuple(zip(*homo)) if homo else ([], []),
        "Non-Homophone": tuple(zip(*non_homo)) if non_homo else ([], []),
    }


def score(refs, hyps, label):
    if not refs:
        return None
    return {
        "label": label, "n": len(refs),
        "WER": word_error_rate(refs, hyps) * 100,
        "CER": character_error_rate(refs, hyps) * 100,
        "PER": phoneme_error_rate(refs, hyps) * 100,
        "BLEU4": bleu4_score(refs, hyps),
        "EM": exact_match(refs, hyps) * 100,
    }


def extended(refs, hyps, label):
    if not refs:
        return None
    wper_panphon = None
    try:
        wper_panphon = ext.weighted_per(refs, hyps, method="panphon") * 100
    except RuntimeError as e:
        print(f"  WPER (panphon) skipped for {label}: {e}")
    return {
        "label": label,
        "sid": ext.sid_breakdown(refs, hyps),
        "aer": ext.allophonic_error_rate(refs, hyps),
        "wper_heuristic": ext.weighted_per(refs, hyps, method="heuristic") * 100,
        "wper_panphon": wper_panphon,
    }


# ── Parallel Stage 2/3 error analysis ────────────────────────────────────────
# analyze_pair is per-row; CMU dict loads inside it. We pass pairs as a list
# and Pool.map forks workers (each gets its own copy of the CMU dict).
def _analyze_chunk(pairs):
    return [analyze_pair(r, h) for r, h in pairs]


def parallel_error_report(refs, hyps, workers: int):
    """Same output shape as error_analysis.error_category_report, but with
    the per-row work distributed across `workers` processes."""
    pairs = list(zip(refs, hyps))
    if workers <= 1 or len(pairs) < 200:
        results = [_analyze_chunk(pairs)]
    else:
        chunk = max(50, len(pairs) // (workers * 4))
        chunks = [pairs[i:i + chunk] for i in range(0, len(pairs), chunk)]
        with Pool(workers) as p:
            results = p.map(_analyze_chunk, chunks)

    flat = [r for chunk_r in results for r in chunk_r]
    totals = Counter()
    cat_counts = Counter()
    stage3_counts = Counter()
    stage3_method_counts = Counter()
    examples = {"Homophone": [], "Near-homophone": [], "Other": []}
    stage3_examples = []
    for ref, hyp, res in zip(refs, hyps, flat):
        for k in ("n_hits", "n_substitutions", "n_deletions", "n_insertions"):
            totals[k] += res[k]
        for s in res["substitutions"]:
            cat_counts[s["category"]] += 1
            if s["category"] in examples and len(examples[s["category"]]) < 5:
                examples[s["category"]].append((ref, hyp, s["ref"], s["hyp"]))
            if s.get("stage3_category"):
                stage3_counts[s["stage3_category"]] += 1
                stage3_method_counts[s["stage3_method"]] += 1
                if len(stage3_examples) < 8:
                    stage3_examples.append({
                        "ref_sentence": ref, "hyp_sentence": hyp,
                        "ref_word": s["ref"], "hyp_word": s["hyp"],
                        "stage3_category":    s["stage3_category"],
                        "stage3_subcategory": s["stage3_subcategory"],
                        "stage3_explanation": s["stage3_explanation"],
                        "stage3_method":      s["stage3_method"],
                    })
    return {
        "overall": {
            "totals": dict(totals),
            "substitution_categories": dict(cat_counts),
            "examples": examples,
            "stage3_categories": dict(stage3_counts),
            "stage3_methods": dict(stage3_method_counts),
            "stage3_examples": stage3_examples,
        }
    }


# ── View file (cheap) ────────────────────────────────────────────────────────
def norm(t):
    t = (t or "").lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def write_view(out_dir, split, n, mode, refs, hyps, is_homo):
    path = os.path.join(out_dir, f"view_{split}_{n}_{mode}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("STATUS | HOMO | PREDICTED | TARGET\n")
        for r, h, m in zip(refs, hyps, is_homo):
            ok = "OK   " if norm(r) == norm(h) else "WRONG"
            f.write(f"{ok} | {'H' if m else '-'} | {h} | {r}\n")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="path to preds_<split>_<n>_<mode>.jsonl")
    ap.add_argument("--mode", required=True, choices=["clean", "raw"])
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel workers for Stage 2/3 (default 1, recommend 8 for train)")
    ap.add_argument("--metrics-out", default=None,
                    help="path to write metrics CSV; default: same dir as jsonl, named metrics_<split>_<n>.csv "
                         "(append one row per mode so both clean+raw land in the same CSV)")
    ap.add_argument("--skip-extended", action="store_true")
    ap.add_argument("--skip-error", action="store_true")
    args = ap.parse_args()

    split, mode, n, out_dir = parse_filename(args.jsonl)

    with open(args.jsonl, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    refs = [r["target"] for r in records]
    hyps = [r["prediction"] for r in records]
    print(f"[{split}/{n}/{mode}] loaded {len(records)} predictions from {args.jsonl}")

    df = data_loader.load_original_phoneme_text_pairs()
    slices = {
        "train": df.iloc[:TRAIN_N],
        "val":   df.iloc[TRAIN_N:TRAIN_N + VAL_N],
        "test":  df.iloc[TRAIN_N + VAL_N:],
    }
    eval_df = slices[split].reset_index(drop=True)
    # Mirror the in-script dedup gate so the homophone mask matches what
    # the predictions were computed against.
    if n != len(eval_df):
        train_sents = set(slices["train"]["sentence"])
        eval_df = eval_df[~eval_df["sentence"].isin(train_sents)].reset_index(drop=True)
    homo_set = set(data_loader.load_homophone_sentences()["sentence"])
    is_homo = [s in homo_set for s in eval_df["sentence"]]
    assert len(is_homo) == len(refs), (
        f"is_homo mask length ({len(is_homo)}) does not match prediction count ({len(refs)}) — "
        f"jsonl was likely written under a different dedup setting than this script is using."
    )

    subsets = stratify(refs, hyps, is_homo)

    # Core metrics
    rows = [r for r in (score(*subsets[label], label) for label in subsets) if r]
    for r in rows:
        r["mode"] = mode
    print(f"\n{'Subset':<16}{'N':>6}{'WER':>9}{'CER':>9}{'PER':>9}{'BLEU4':>8}{'EM':>8}")
    for r in rows:
        print(f"{r['label']:<16}{r['n']:>6}{r['WER']:>8.2f}%{r['CER']:>8.2f}%"
              f"{r['PER']:>8.2f}%{r['BLEU4']:>8.4f}{r['EM']:>7.2f}%")

    # View
    write_view(out_dir, split, n, mode, refs, hyps, is_homo)

    # Extended
    if not args.skip_extended:
        print(f"\n  Running extended metrics (SID/AER/WPER)...", flush=True)
        ext_rows = [e for e in (extended(*subsets[label], label) for label in subsets) if e]
        for e in ext_rows:
            aer = e["aer"]
            wper_p = f"{e['wper_panphon']:.2f}%" if e["wper_panphon"] is not None else "skipped"
            print(f"  [{e['label']}] WPER heuristic={e['wper_heuristic']:.2f}%  panphon={wper_p}")
            print(f"      AER: place={aer['place_pct']:.1f}%  manner={aer['manner_pct']:.1f}%  "
                  f"voicing={aer['voicing_pct']:.1f}%  (of {aer['n_classified']} classified substitutions)")
        with open(os.path.join(out_dir, f"extended_{split}_{n}_{mode}.json"), "w", encoding="utf-8") as f:
            json.dump(ext_rows, f, indent=2, ensure_ascii=False, default=str)

    # Error analysis (parallel)
    if not args.skip_error:
        print(f"\n  Running Stage 2/3 error analysis with {args.workers} workers...", flush=True)
        import time
        t0 = time.time()
        report = parallel_error_report(refs, hyps, workers=args.workers)
        print(f"  Stage 2/3 done in {time.time()-t0:.1f}s "
              f"(substitutions: {report['overall']['totals']['n_substitutions']})")
        with open(os.path.join(out_dir, f"errors_{split}_{n}_{mode}.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # Metrics CSV: append this mode's rows to the existing file (or create).
    metrics_path = args.metrics_out or os.path.join(out_dir, f"metrics_{split}_{n}.csv")
    fieldnames = ["mode", "label", "n", "WER", "CER", "PER", "BLEU4", "EM"]
    existing_rows = []
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r", newline="") as f:
            existing_rows = list(csv.DictReader(f))
    existing_rows.extend(rows)
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(existing_rows)
    print(f"\nMetrics -> {metrics_path}")


if __name__ == "__main__":
    main()