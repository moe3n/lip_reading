"""
Stage 3 Option 5 (LLM judge) error analysis — runs the full error-pattern
report (Stage 2 + Stage 3 Options 3+5) on any predictions.csv with
phonemes/target/prediction/is_homophone columns — the format both
p2t_lora/dryrun.py and zero-shot/run_baseline.py produce.

Uses a genuinely instruct-tuned model as judge (meta-llama/Llama-3.2-3B-Instruct
by default) — NOT the small Qwen stand-in, which llm_judge.py's own
documented empirical test found defaults to "Semantic" for everything
regardless of the actual error.

Usage:
    python analyze_errors.py [path/to/predictions.csv]
Defaults to p2t_lora_checkpoints/predictions.csv. Writes error_report.json
and error_report.png next to the input CSV (overwriting a prior Option-3-only
pass, since this is the same report with the judge filled in — not a merge).
"""

import csv
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from p2t_lora.model import load_tokenizer                              # noqa: E402
from p2t_lora.evaluation.error_analysis import (                       # noqa: E402
    error_category_report, print_error_report, plot_error_report,
)

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")


def bnb_compute_dtype() -> torch.dtype:
    # bf16 needs compute capability >= 8.0 (Ampere+); fp16 on older GPUs (e.g. Pascal).
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.bfloat16
    return torch.float16


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("p2t_lora_checkpoints", "predictions.csv")
    out_dir = os.path.dirname(csv_path) or "."

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    refs = [r["target"] for r in rows]
    hyps = [r["prediction"] for r in rows]
    homo_mask = [r["is_homophone"] == "True" for r in rows]
    print(f"Loaded {len(rows)} rows from {csv_path}")

    print(f"Loading judge model ({JUDGE_MODEL})...")
    tokenizer = load_tokenizer(JUDGE_MODEL)
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bnb_compute_dtype(),
    )
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, quantization_config=quant, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))  # no-op unless load_tokenizer added a [PAD] token
    model.eval()
    print(f"Loaded {JUDGE_MODEL} (4-bit, compute dtype {bnb_compute_dtype()})")

    report = error_category_report(refs, hyps, homo_mask, tokenizer=tokenizer, model=model, use_llm=True)
    print_error_report(report, title=f"{JUDGE_MODEL} judge — error pattern analysis (Stage 2+3, full)")

    out_json = os.path.join(out_dir, "error_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    out_png = os.path.join(out_dir, "error_report.png")
    plot_error_report(report, out_png, title=f"Substitution error categories (judge: {JUDGE_MODEL})")

    print(f"Saved: {out_json}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
