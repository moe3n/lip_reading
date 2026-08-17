"""
Regenerate the dedup validation predictions from the existing epoch_3 checkpoint
using beam search (width 5) instead of greedy, so the dedup-vs-full comparison
isolates decoding strategy as the only variable.

No retraining. Loads the already-trained adapter in p2t_lora_checkpoints_dedup/epoch_3.

Usage (from repo root, on the uni GPU box):
    python regen_dedup_beam5.py
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

CHECKPOINT_ROOT = "p2t_lora_checkpoints_dedup"
EPOCH_DIR       = os.path.join(CHECKPOINT_ROOT, "epoch_3")
BASE_MODEL      = "meta-llama/Llama-3.2-3B"
TRAIN_N, VAL_N  = 45839, 1082


def main():
    print("Loading corpus ...")
    full_df = data_loader.load_original_phoneme_text_pairs()
    train_df = full_df.iloc[:TRAIN_N].reset_index(drop=True)
    val_df   = full_df.iloc[TRAIN_N:TRAIN_N + VAL_N].reset_index(drop=True)

    train_sentences = set(train_df["sentence"])
    val_clean = val_df[~val_df["sentence"].isin(train_sentences)].reset_index(drop=True)
    print(f"Clean val rows: {len(val_clean)} (matches the {949} used for the greedy dedup run)")

    homo_df, _ = data_loader.load_stratified_split(full_df)
    homo_set = set(homo_df["sentence"])

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
    # Training added a [PAD] token (Llama-3 has none) and resized embeddings,
    # so the adapter checkpoint holds a 128,257-row embed/lm_head. Match that
    # here before loading the adapter, or PeftModel.from_pretrained raises a
    # size-mismatch on embed_tokens and lm_head.
    if len(tokenizer) != base.get_input_embeddings().weight.shape[0]:
        base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, EPOCH_DIR)
    model.eval()
    device = next(model.parameters()).device
    print(f"Model on {device}")

    print(f"\nGenerating on {len(val_clean)} examples with beam search (width 5) ...")
    refs, hyps, homo_mask = [], [], []
    for i, (_, row) in enumerate(val_clean.iterrows(), 1):
        if i % 100 == 0:
            print(f"  {i}/{len(val_clean)}", flush=True)
        prompt = f"Phonemes: {row['phonemes']}\nText:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=34,
                do_sample=False,
                num_beams=5,
                early_stopping=True,
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

    pd.DataFrame({
        "target": refs, "prediction": hyps, "is_homophone": homo_mask,
    }).to_csv(os.path.join(CHECKPOINT_ROOT, "predictions_beam5.csv"), index=False)

    eval_results = stratified_evaluate(refs, hyps, homo_mask)
    print_results(eval_results, title="LoRA epoch_3 (dedup val) — beam-5 decoding")
    save_results(eval_results, os.path.join(CHECKPOINT_ROOT, "metrics_beam5.csv"), model_name=BASE_MODEL)

    print(f"\nDone. predictions_beam5.csv and metrics_beam5.csv written to {CHECKPOINT_ROOT}/")


if __name__ == "__main__":
    main()
