"""Evaluate a LoRA checkpoint on the held-out LRS2 test split (1,243 rows), reporting standard and deduplicated metrics."""

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

CHECKPOINT_ROOT = os.environ.get("TEST_CHECKPOINT", "p2t_lora_checkpoints_dedup")
EPOCH_DIR       = os.path.join(CHECKPOINT_ROOT, "epoch_3")
BASE_MODEL      = "meta-llama/Llama-3.2-3B"
OUT_DIR         = os.path.join(CHECKPOINT_ROOT, "test_eval")
NUM_BEAMS       = int(os.environ.get("TEST_NUM_BEAMS", "5"))
TRAIN_N, VAL_N  = 45839, 1082

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading corpus ...")
    full_df = data_loader.load_original_phoneme_text_pairs()
    train_sentences = set(full_df.iloc[:TRAIN_N]["sentence"])
    homo_df, _ = data_loader.load_stratified_split(full_df)
    homo_set = set(homo_df["sentence"])

    test_df = full_df.iloc[TRAIN_N + VAL_N:].reset_index(drop=True)
    dup_flag = test_df["sentence"].isin(train_sentences)
    dup_count = int(dup_flag.sum())
    print(f"Held-out test: {len(test_df)} rows total | {dup_count} appear in training "
          f"| {len(test_df) - dup_count} deduplicated")

    print(f"Loading {BASE_MODEL} (4-bit NF4) + adapter from {EPOCH_DIR}  (beam={NUM_BEAMS}) ...")
    patch_bnb_safe_to()
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_ROOT)
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_cfg,
                                                device_map="auto")
    if len(tokenizer) != base.get_input_embeddings().weight.shape[0]:
        base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, EPOCH_DIR)
    model.eval()
    device = next(model.parameters()).device

    print(f"\nGenerating on all {len(test_df)} test rows ...")
    refs, hyps, homo_mask = [], [], []
    for i, (_, row) in enumerate(test_df.iterrows(), 1):
        if i % 100 == 0:
            print(f"  {i}/{len(test_df)}", flush=True)
        inputs = tokenizer(f"Phonemes: {row['phonemes']}\nText:", return_tensors="pt").to(device)
        kwargs = dict(max_new_tokens=34, do_sample=False,
                      pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id,
                      repetition_penalty=1.3, no_repeat_ngram_size=3)
        if NUM_BEAMS > 1:
            kwargs.update(num_beams=NUM_BEAMS, early_stopping=True)
        with torch.no_grad():
            gen = model.generate(**inputs, **kwargs)
        decoded = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip().split("\n", 1)[0].strip()
        refs.append(row["sentence"]); hyps.append(decoded)
        homo_mask.append(row["sentence"] in homo_set)

    pd.DataFrame({"target": refs, "prediction": hyps, "is_homophone": homo_mask,
                  "in_training": list(dup_flag)}).to_csv(
        os.path.join(OUT_DIR, "predictions.csv"), index=False)

    print("\n" + "=" * 60 + "\n  STANDARD TEST SET (all rows, comparable to published split)")
    std = stratified_evaluate(refs, hyps, homo_mask)
    print_results(std, title=f"{CHECKPOINT_ROOT}: standard test set")
    save_results(std, os.path.join(OUT_DIR, "metrics_standard.csv"), model_name=BASE_MODEL)

    keep = [not d for d in dup_flag]
    dref = [r for r, k in zip(refs, keep) if k]
    dhyp = [h for h, k in zip(hyps, keep) if k]
    dhom = [m for m, k in zip(homo_mask, keep) if k]
    print("\n" + "=" * 60 + "\n  DEDUPLICATED TEST SET (leakage-free figure)")
    ded = stratified_evaluate(dref, dhyp, dhom)
    print_results(ded, title=f"{CHECKPOINT_ROOT}: deduplicated test set")
    save_results(ded, os.path.join(OUT_DIR, "metrics_dedup.csv"), model_name=BASE_MODEL)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"checkpoint": EPOCH_DIR, "num_beams": NUM_BEAMS,
                   "test_total": len(test_df), "in_training": dup_count,
                   "deduplicated_rows": len(test_df) - dup_count,
                   "standard_EM": round(std["overall"]["Exact_Match"] * 100, 2),
                   "dedup_EM": round(ded["overall"]["Exact_Match"] * 100, 2)}, f, indent=2)
    print(f"\nDone. Results in {OUT_DIR}/")

if __name__ == "__main__":
    main()
