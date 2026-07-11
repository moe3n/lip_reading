"""
Detailed error-pattern analysis for the fine-tuned p2t_lora model, computed
from its already-saved predictions.csv (src/p2t_lora/dryrun.py's validation
generations) -- no retraining, no GPU, just scoring cached text.

Same shape as comparison/recompute_mira.py, so the LoRA and prompting-only
legs of the comparison produce directly comparable reports.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pandas as pd  # noqa: E402

from p2t_lora.evaluation.error_analysis import error_category_report, print_error_report  # noqa: E402
from p2t_lora.evaluation import extended_metrics as ext  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_CSV = os.path.join(os.path.dirname(HERE), "p2t_lora_checkpoints", "predictions.csv")
OUT_DIR = os.path.join(HERE, "results")


def main():
    df = pd.read_csv(PREDICTIONS_CSV)
    refs = df["target"].tolist()
    hyps = df["prediction"].fillna("").tolist()
    is_homo = df["is_homophone"].astype(bool).tolist()
    print(f"Loaded {len(df)} rows from {PREDICTIONS_CSV}")

    print("\nRunning error pattern analysis (Stage 2/3)...", flush=True)
    report = error_category_report(refs, hyps, homo_mask=is_homo)
    print_error_report(report, title="p2t_lora fine-tuned -- error pattern analysis")

    print("Running extended metrics (SID/AER/WPER)...", flush=True)
    sid = ext.sid_breakdown(refs, hyps)
    aer = ext.allophonic_error_rate(refs, hyps)
    wper_h = ext.weighted_per(refs, hyps, method="heuristic") * 100
    try:
        wper_p = ext.weighted_per(refs, hyps, method="panphon") * 100
    except RuntimeError as e:
        wper_p = None
        print(f"WPER (panphon) skipped: {e}")

    print("SID:", sid)
    print("AER:", aer)
    print(f"WPER (heuristic): {wper_h:.2f}%")
    if wper_p is not None:
        print(f"WPER (panphon): {wper_p:.2f}%")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "source": "p2t_lora fine-tuned (Llama-3.2-3B + LoRA, no contrastive)",
        "n": len(df),
        "error_report": report,
        "sid": sid,
        "aer": aer,
        "wper_heuristic_pct": wper_h,
        "wper_panphon_pct": wper_p,
    }
    out_path = os.path.join(OUT_DIR, "lora_error_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
