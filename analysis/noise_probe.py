"""
Noise sensitivity probe: how much does the fine-tuned model degrade when the
phoneme input is corrupted?

The model was trained only on clean ground-truth phonemes. In a real lip-reading
pipeline the phonemes come from a visual front-end that makes mistakes, so this
measures what happens when they do. Nothing is retrained: same epoch_3 adapter,
same rows, same beam-5 decoding as the headline result. The ONLY thing that
changes between conditions is the input phoneme string.

Three corruption types, applied at several rates, plus a clean control on the
identical subsample so every comparison is internally consistent:

    substitute  swap a phoneme for a different one from the corpus inventory
    delete      drop a phoneme
    insert      add a spurious phoneme

Reading the output: if metrics fall off a cliff between 5% and 10% noise, the
model is brittle and noise-augmented training is worth doing. If it degrades
gracefully, the fine-tune already generalises and augmentation buys less.

Config via env vars:
    NOISE_N_ROWS   how many val rows to probe (default 300; 949 = all, ~3x slower)
    NOISE_RATES    comma-separated corruption rates (default 0.05,0.10,0.20)
    NOISE_SEED     RNG seed (default 42)

Usage (uni GPU box):
    python analysis/noise_probe.py
"""

import os
import sys
import json
import random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from p2t_lora.data import loader as data_loader
from p2t_lora.evaluation.metrics import stratified_evaluate
from p2t_lora.augmentation.phoneme_noise import corrupt, phoneme_inventory
from p2t_lora.model import patch_bnb_safe_to

# Which trained model to probe. Both arms of the ablation need probing:
#   NOISE_CHECKPOINT=p2t_lora_checkpoints_dedup  -> clean-trained baseline
#   NOISE_CHECKPOINT=p2t_lora_checkpoints_noise  -> noise-augmented model
# Output goes to its own directory per checkpoint so the two never overwrite.
CHECKPOINT_ROOT = os.environ.get("NOISE_CHECKPOINT", "p2t_lora_checkpoints_dedup")
EPOCH_DIR       = os.path.join(CHECKPOINT_ROOT, "epoch_3")
BASE_MODEL      = "meta-llama/Llama-3.2-3B"
OUT_DIR         = os.environ.get(
    "NOISE_OUT_DIR",
    os.path.join("analysis", f"noise_probe_{os.path.basename(CHECKPOINT_ROOT)}"),
)
TRAIN_N, VAL_N  = 45839, 1082

N_ROWS = int(os.environ.get("NOISE_N_ROWS", "300"))
RATES  = [float(r) for r in os.environ.get("NOISE_RATES", "0.05,0.10,0.20").split(",")]
SEED   = int(os.environ.get("NOISE_SEED", "42"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Written before generation, not after, so a run that dies partway still
    # says what it was doing. A directory holding config.json but no
    # summary.csv means generation did not finish.
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump({"n_rows": N_ROWS, "rates": RATES, "seed": SEED,
                   "checkpoint": EPOCH_DIR, "decoding": "beam-5"}, f, indent=2)

    print(f"Probing checkpoint: {EPOCH_DIR}")
    print(f"Output directory:   {OUT_DIR}")
    print("Loading corpus ...")
    full_df = data_loader.load_original_phoneme_text_pairs()
    train_df = full_df.iloc[:TRAIN_N].reset_index(drop=True)
    val_df   = full_df.iloc[TRAIN_N:TRAIN_N + VAL_N].reset_index(drop=True)
    val_clean = val_df[~val_df["sentence"].isin(set(train_df["sentence"]))].reset_index(drop=True)

    # Fixed-seed subsample so every condition sees the exact same sentences.
    probe = val_clean.sample(n=min(N_ROWS, len(val_clean)), random_state=SEED).reset_index(drop=True)

    homo_df, _ = data_loader.load_stratified_split(full_df)
    homo_set = set(homo_df["sentence"])
    homo_mask = [s in homo_set for s in probe["sentence"]]

    # Phoneme inventory taken from the corpus itself rather than hardcoded, so
    # substituted/inserted phonemes are always ones the model has actually seen.
    inventory = phoneme_inventory(full_df["phonemes"])
    print(f"Probing {len(probe)} rows | {len(inventory)} phonemes in inventory | seed {SEED}")

    print(f"Loading tokenizer from {CHECKPOINT_ROOT} ...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_ROOT)

    print(f"Loading {BASE_MODEL} (4-bit NF4) + adapter from {EPOCH_DIR} ...")
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

    def generate(phoneme_strings):
        hyps = []
        for i, ph in enumerate(phoneme_strings, 1):
            if i % 100 == 0:
                print(f"    {i}/{len(phoneme_strings)}", flush=True)
            inputs = tokenizer(f"Phonemes: {ph}\nText:", return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=34, do_sample=False,
                    num_beams=5, early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.3, no_repeat_ngram_size=3,
                )
            out = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            hyps.append(out.split("\n", 1)[0].strip())
        return hyps

    refs = list(probe["sentence"])
    conditions = [("clean", 0.0)] + [(k, r) for k in ("substitute", "delete", "insert") for r in RATES]
    summary = []

    for kind, rate in conditions:
        tag = "clean" if kind == "clean" else f"{kind}_{int(rate * 100)}pct"
        print(f"\n[{tag}] generating ...")
        rng = random.Random(SEED)  # reset per condition: same rows, reproducible corruption
        if kind == "clean":
            inputs_ph = list(probe["phonemes"])
        else:
            inputs_ph = [corrupt(p, kind, rate, rng, inventory) for p in probe["phonemes"]]

        hyps = generate(inputs_ph)
        res = stratified_evaluate(refs, hyps, homo_mask)["overall"]
        summary.append({
            "condition": tag, "kind": kind, "rate": rate,
            "WER": res["WER"] * 100, "CER": res["CER"] * 100,
            "ExactMatch": res["Exact_Match"] * 100,
        })
        print(f"    WER {res['WER'] * 100:.2f}%   EM {res['Exact_Match'] * 100:.2f}%")

        pd.DataFrame({
            "target": refs, "input_phonemes": inputs_ph, "prediction": hyps,
        }).to_csv(os.path.join(OUT_DIR, f"predictions_{tag}.csv"), index=False)

    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    print("\n" + "=" * 60)
    print(f"  Noise sensitivity — {CHECKPOINT_ROOT}")
    print(f"  {len(probe)} rows, beam-5, epoch_3")
    print("=" * 60)
    print(f"  {'Condition':<20}{'WER':>9}{'CER':>9}{'EM':>9}")
    print("  " + "-" * 45)
    for r in summary:
        print(f"  {r['condition']:<20}{r['WER']:>8.2f}%{r['CER']:>8.2f}%{r['ExactMatch']:>8.2f}%")
    print("=" * 60)

    clean_em = next(r["ExactMatch"] for r in summary if r["condition"] == "clean")
    worst = min(summary, key=lambda r: r["ExactMatch"])
    print(f"\n  Clean control: {clean_em:.2f}% EM")
    print(f"  Worst case ({worst['condition']}): {worst['ExactMatch']:.2f}% EM "
          f"({clean_em - worst['ExactMatch']:.2f} points below clean)")
    print(f"\nDone. Per-condition predictions + summary.csv in {OUT_DIR}/")


if __name__ == "__main__":
    main()
