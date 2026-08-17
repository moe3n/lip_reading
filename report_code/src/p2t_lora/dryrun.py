"""Train Llama-3.2-3B with QLoRA on LRS2 phoneme-to-text (plain causal-LM fine-tuning); configured via CPT_* environment variables."""

import os
import sys
import time
import json
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2t_lora.data import loader as data_loader
from p2t_lora.augmentation.phoneme_noise import corrupt_random, phoneme_inventory
from p2t_lora.model import (
    load_tokenizer, load_model_with_lora, MODEL_NAME_DRYRUN, DEVICE, USE_4BIT,
)
from p2t_lora.evaluation.metrics import stratified_evaluate, print_results, save_results
from p2t_lora.evaluation.error_analysis import (
    error_category_report, print_error_report, plot_error_report,
)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)

def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val else default

def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val else default

def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    return val.strip().lower() in ("1", "true", "yes", "on") if val else default


CFG = {
    "model_name":        _env_str("CPT_MODEL_NAME", MODEL_NAME_DRYRUN),
    "n_homophone":        _env_int("CPT_N_HOMOPHONE", 130),
    "n_non_homophone":    _env_int("CPT_N_NON_HOMOPHONE", 70),
    "n_total":            _env_int("CPT_N_TOTAL", 0),
    "seq_split":          _env_bool("CPT_SEQ_SPLIT", False),
    "max_input_len":      _env_int("CPT_MAX_INPUT_LEN", 96),
    "max_target_len":     _env_int("CPT_MAX_TARGET_LEN", 32),
    "lora_r":             _env_int("CPT_LORA_R", 8),
    "lora_alpha":         _env_int("CPT_LORA_ALPHA", 16),
    "lora_dropout":       _env_float("CPT_LORA_DROPOUT", 0.1),
    "epochs":             _env_int("CPT_EPOCHS", 2),
    "num_beams":          _env_int("CPT_NUM_BEAMS", 1),
    "noise_prob":         _env_float("CPT_NOISE_PROB", 0.0),
    "noise_rate_min":     _env_float("CPT_NOISE_RATE_MIN", 0.05),
    "noise_rate_max":     _env_float("CPT_NOISE_RATE_MAX", 0.15),
    "noise_seed":         _env_int("CPT_NOISE_SEED", 42),
    "seed":               _env_int("CPT_SEED", 42),
    "batch_size":         _env_int("CPT_BATCH_SIZE", 2),
    "grad_accumulation":  _env_int("CPT_GRAD_ACCUM", 2),
    "learning_rate":      _env_float("CPT_LEARNING_RATE", 2e-4),
    "warmup_steps":       _env_int("CPT_WARMUP_STEPS", 2),
    "checkpoint_dir":     _env_str("CPT_CHECKPOINT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "p2t_lora_checkpoints")),
    "llm_error_judge":    _env_bool("CPT_LLM_ERROR_JUDGE", False),
}


class PhonemeTextDataset(Dataset):
    """Tokenise phoneme-to-text pairs, mask the prompt in the labels, and optionally corrupt the phonemes when `noise` is given."""

    def __init__(self, df, tokenizer, homo_set, max_input=96, max_target=32, noise=None):
        self.samples = []
        max_len = max_input + max_target
        rng = random.Random(noise["seed"]) if noise else None
        self.n_corrupted = 0

        for _, row in df.iterrows():
            sentence = row["sentence"]
            phonemes = row["phonemes"]
            is_homo = sentence in homo_set

            if noise:
                noisy = corrupt_random(
                    phonemes, rng, noise["inventory"],
                    noise["prob"], noise["rate_min"], noise["rate_max"],
                )
                if noisy != phonemes:
                    self.n_corrupted += 1
                phonemes = noisy

            prefix = f"Phonemes: {phonemes}\nText:"
            full_text = f"{prefix} {sentence}{tokenizer.eos_token}"

            prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
            prefix_len = min(len(prefix_ids), max_len - 1)

            enc = tokenizer(
                full_text, max_length=max_len, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            labels = input_ids.clone()
            labels[:prefix_len] = -100
            labels[attention_mask == 0] = -100

            self.samples.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "labels":         labels,
                "is_homophone":   torch.tensor(is_homo, dtype=torch.bool),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def compute_loss(model, batch):
    """Cross-entropy loss on the sentence tokens; the prompt tokens are masked out in `labels`."""
    input_ids      = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    labels         = batch["labels"].to(DEVICE)
    return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss


def build_dryrun_dataframes():
    """Load the LRS2 pairs and return train/val dataframes (official 45,839/1,082 split when CPT_SEQ_SPLIT is set)."""
    full_df = data_loader.load_original_phoneme_text_pairs()
    homo_df, non_homo_df = data_loader.load_stratified_split(full_df)
    homo_set = set(homo_df["sentence"])

    if CFG["seq_split"]:
        TRAIN_N, VAL_N = 45839, 1082
        train_df = full_df.iloc[:TRAIN_N].reset_index(drop=True)
        val_df   = full_df.iloc[TRAIN_N:TRAIN_N + VAL_N].reset_index(drop=True)
        train_sentences = set(train_df["sentence"])
        val_df = val_df[~val_df["sentence"].isin(train_sentences)].reset_index(drop=True)
        print(f"  Dedup: {VAL_N} val rows -> {len(val_df)} after removing training duplicates")
        return train_df, val_df, homo_set

    if CFG["n_total"]:
        df = full_df.sample(n=min(CFG["n_total"], len(full_df)), random_state=42).reset_index(drop=True)
    else:
        df_homo_sub = homo_df.head(CFG["n_homophone"]).copy()
        df_non_sub = non_homo_df.head(CFG["n_non_homophone"]).copy()
        df = pd.concat([df_homo_sub, df_non_sub], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split = max(1, int(len(df) * 0.8))
    return df[:split].reset_index(drop=True), df[split:].reset_index(drop=True), homo_set


def main():
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG["seed"])
    sentences_desc = (f"{CFG['n_total']} (unstratified)" if CFG["n_total"]
                       else f"{CFG['n_homophone']} homophone + {CFG['n_non_homophone']} non-homophone")
    print(f"Model: {CFG['model_name']}  |  Sentences: {sentences_desc}  |  "
          f"LoRA r={CFG['lora_r']}  |  Epochs: {CFG['epochs']}")
    print(f"Device: {DEVICE}  |  4-bit QLoRA active: {USE_4BIT}")

    print("\nLoading data (real LRS2 sentences + phoneme transcriptions, original order)...")
    df_tr, df_val, homo_set = build_dryrun_dataframes()
    print(f"  Train: {len(df_tr)}  |  Val: {len(df_val)}")

    print(f"\nLoading tokenizer + model ({CFG['model_name']})...")
    tokenizer = load_tokenizer(CFG["model_name"])
    model = load_model_with_lora(
        CFG["model_name"], CFG["lora_r"], CFG["lora_alpha"], CFG["lora_dropout"],
        tokenizer=tokenizer,
    )

    noise_cfg = None
    if CFG["noise_prob"] > 0:
        full_df = data_loader.load_original_phoneme_text_pairs()
        noise_cfg = {
            "prob":      CFG["noise_prob"],
            "rate_min":  CFG["noise_rate_min"],
            "rate_max":  CFG["noise_rate_max"],
            "seed":      CFG["noise_seed"],
            "inventory": phoneme_inventory(full_df["phonemes"]),
        }
        print(f"\nNoise augmentation ON: {CFG['noise_prob']:.0%} of training rows, "
              f"rate {CFG['noise_rate_min']:.0%}-{CFG['noise_rate_max']:.0%}, "
              f"seed {CFG['noise_seed']}  (validation stays clean)")

    train_ds = PhonemeTextDataset(df_tr, tokenizer, homo_set, CFG["max_input_len"],
                                  CFG["max_target_len"], noise=noise_cfg)
    val_ds = PhonemeTextDataset(df_val, tokenizer, homo_set, CFG["max_input_len"], CFG["max_target_len"])
    if noise_cfg:
        print(f"  {train_ds.n_corrupted}/{len(train_ds)} training examples actually corrupted")
    train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False)
    print(f"  Train batches: {len(train_dl)}  |  Val batches: {len(val_dl)}")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                       lr=CFG["learning_rate"], weight_decay=0.01)
    total_steps = max(1, len(train_dl) * CFG["epochs"] // CFG["grad_accumulation"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=CFG["warmup_steps"], num_training_steps=total_steps,
    )

    print(f"\nTraining ({CFG['epochs']} epochs, {len(df_tr) + len(df_val)} sentences)...")
    print("-" * 60)
    history = []
    n_tr_total = len(train_dl)
    for epoch in range(CFG["epochs"]):
        model.train()
        train_total = 0.0
        optimizer.zero_grad()
        t_epoch_start = time.time()
        for step, batch in enumerate(train_dl):
            t_step_start = time.time()
            loss = compute_loss(model, batch)
            (loss / CFG["grad_accumulation"]).backward()
            train_total += loss.item()
            if (step + 1) % CFG["grad_accumulation"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            print(f"  epoch {epoch + 1} step {step + 1}/{n_tr_total}: "
                  f"loss={loss.item():.4f}  ({time.time() - t_step_start:.1f}s/step)",
                  flush=True)
        print(f"  -> epoch {epoch + 1} train pass took {time.time() - t_epoch_start:.1f}s total", flush=True)

        n_tr = len(train_dl)
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in val_dl:
                val_total += compute_loss(model, batch).item()
        n_val = max(1, len(val_dl))

        ep_log = {"epoch": epoch + 1, "train_loss": train_total / n_tr, "val_loss": val_total / n_val}
        history.append(ep_log)
        print(f"Epoch {ep_log['epoch']}: train_loss={ep_log['train_loss']:.4f}  |  "
              f"val_loss={ep_log['val_loss']:.4f}")

        model.save_pretrained(os.path.join(CFG["checkpoint_dir"], f"epoch_{epoch + 1}"))
        with open(os.path.join(CFG["checkpoint_dir"], "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"  (epoch {epoch + 1} adapter saved to epoch_{epoch + 1}/)", flush=True)

    print("-" * 60)
    model.save_pretrained(CFG["checkpoint_dir"])
    tokenizer.save_pretrained(CFG["checkpoint_dir"])
    print(f"Checkpoint saved to: {CFG['checkpoint_dir']}")

    print(f"\nGenerating on all {len(df_val)} validation examples...")
    model.eval()
    all_phonemes, all_refs, all_hyps, homo_mask = [], [], [], []
    for _, row in df_val.iterrows():
        prompt = f"Phonemes: {row['phonemes']}\nText:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen_kwargs = dict(max_new_tokens=34, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id,
                              eos_token_id=tokenizer.eos_token_id,
                              repetition_penalty=1.3,
                              no_repeat_ngram_size=3)
            if CFG["num_beams"] > 1:
                gen_kwargs.update(num_beams=CFG["num_beams"], early_stopping=True)
            gen = model.generate(**inputs, **gen_kwargs)
        decoded = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        decoded = decoded.split("\n", 1)[0].strip()
        all_phonemes.append(row["phonemes"])
        all_refs.append(row["sentence"])
        all_hyps.append(decoded)
        homo_mask.append(row["sentence"] in homo_set)

    print("\nSample generations (first 3):")
    for ref, hyp in zip(all_refs[:3], all_hyps[:3]):
        print(f"  Ref : {ref}")
        print(f"  Gen : {hyp}")
        print()

    predictions_path = os.path.join(CFG["checkpoint_dir"], "predictions.csv")
    pd.DataFrame({
        "phonemes": all_phonemes, "target": all_refs,
        "prediction": all_hyps, "is_homophone": homo_mask,
    }).to_csv(predictions_path, index=False)
    print(f"Per-row phonemes/target/prediction saved to: {predictions_path}")

    eval_results = stratified_evaluate(all_refs, all_hyps, homo_mask)
    print_results(eval_results, title=f"{CFG['model_name']} dry run: generation metrics")

    metrics_csv = os.path.join(CFG["checkpoint_dir"], "metrics_log.csv")
    save_results(eval_results, metrics_csv, model_name=CFG["model_name"])

    error_report = error_category_report(
        all_refs, all_hyps, homo_mask,
        tokenizer=tokenizer, model=model, use_llm=CFG["llm_error_judge"],
    )
    print_error_report(error_report, title=f"{CFG['model_name']} dry run: error pattern analysis")

    error_report_path = os.path.join(CFG["checkpoint_dir"], "error_report.json")
    with open(error_report_path, "w", encoding="utf-8") as f:
        json.dump(error_report, f, indent=2, ensure_ascii=False, default=str)

    error_chart_path = os.path.join(CFG["checkpoint_dir"], "error_report.png")
    plot_error_report(error_report, error_chart_path,
                       title=f"{CFG['model_name']}: substitution error categories")
    print(f"Detailed error pattern analysis saved to: {error_report_path}")
    print(f"Error pattern chart saved to: {error_chart_path}")

    return history


if __name__ == "__main__":
    main()
