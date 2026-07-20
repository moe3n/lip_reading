"""
Evaluate the full-corpus LoRA checkpoint (epoch_3) on the clean test split.

Clean test = the 1,243 held-out test rows (corpus positions 46,922-48,164)
minus any sentence that appears verbatim in the 45,839 training rows.
The original CSV is never modified; the filter happens in memory.

Output: p2t_lora_checkpoints_full/test_eval/
    predictions.csv  -- target / prediction / is_homophone per row
    metrics.csv      -- WER / CER / EM / BLEU overall + homophone split
    summary.json     -- row counts (total / duplicates removed / clean)

Usage (from repo root, on the uni GPU box after huggingface-cli login):
    python eval_test_clean.py
"""

import os
import sys
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from p2t_lora.data import loader as data_loader
from p2t_lora.evaluation.metrics import stratified_evaluate, print_results, save_results
from p2t_lora.model import patch_bnb_safe_to

CHECKPOINT_ROOT = "p2t_lora_checkpoints_full"
EPOCH_DIR       = os.path.join(CHECKPOINT_ROOT, "epoch_3")
BASE_MODEL      = "meta-llama/Llama-3.2-3B"
OUT_DIR         = os.path.join(CHECKPOINT_ROOT, "test_eval")
TRAIN_N, VAL_N  = 45839, 1082


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading corpus ...")
    full_df = data_loader.load_original_phoneme_text_pairs()

    # Sentences are already uppercase-normalised by load_original_phoneme_text_pairs
    train_sentences = set(full_df.iloc[:TRAIN_N]["sentence"])

    homo_df, _ = data_loader.load_stratified_split(full_df)
    homo_set = set(homo_df["sentence"])

    test_df = full_df.iloc[TRAIN_N + VAL_N:].reset_index(drop=True)
    clean_mask = ~test_df["sentence"].isin(train_sentences)
    test_clean = test_df[clean_mask].reset_index(drop=True)
    dup_count = int((~clean_mask).sum())
    print(f"Test rows: {len(test_df)} total | {dup_count} duplicates removed | "
          f"{len(test_clean)} clean rows")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading tokenizer from {CHECKPOINT_ROOT} ...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_ROOT)

    print(f"Loading {BASE_MODEL} (4-bit NF4) + LoRA adapter from {EPOCH_DIR} ...")
    patch_bnb_safe_to()   # needed when CUDA_VISIBLE_DEVICES pins one GPU
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_cfg, device_map="auto",
    )
    if len(tokenizer) != base.get_input_embeddings().weight.shape[0]:
        base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, EPOCH_DIR)
    model.eval()
    device = next(model.parameters()).device
    print(f"Model on {device}")

    # ── Generate ──────────────────────────────────────────────────────────────
    print(f"\nGenerating on {len(test_clean)} examples ...")
    refs, hyps, homo_mask = [], [], []
    for i, (_, row) in enumerate(test_clean.iterrows(), 1):
        if i % 100 == 0:
            print(f"  {i}/{len(test_clean)}", flush=True)
        prompt = f"Phonemes: {row['phonemes']}\nText:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=34,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
        decoded = tokenizer.decode(
            gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        decoded = decoded.split("\n", 1)[0].strip()
        refs.append(row["sentence"])
        hyps.append(decoded)
        homo_mask.append(row["sentence"] in homo_set)

    # ── Save + Score ──────────────────────────────────────────────────────────
    pd.DataFrame({
        "target": refs, "prediction": hyps, "is_homophone": homo_mask,
    }).to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)

    eval_results = stratified_evaluate(refs, hyps, homo_mask)
    print_results(eval_results, title="LoRA epoch_3 — clean test set")
    save_results(eval_results, os.path.join(OUT_DIR, "metrics.csv"), model_name=BASE_MODEL)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "checkpoint": EPOCH_DIR,
            "test_total": len(test_df),
            "duplicates_removed": dup_count,
            "clean_test_rows": len(test_clean),
        }, f, indent=2)

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
