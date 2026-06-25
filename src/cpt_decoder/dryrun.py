"""
CPT Decoder — Llama 3.2:3B + QLoRA Dry Run (Phase 1 architecture port)
=========================================================================
Ports CPT_Decoder_Phase2_Mini.ipynb (Flan-T5-small encoder-decoder + LoRA)
to a decoder-only causal LM + QLoRA, per the dissertation's stated approach
(Project_Status_Summary.md ss2/ss7).

WHAT CHANGED FROM THE FLAN-T5 PROTOTYPE, AND WHY
--------------------------------------------------
1. Model loading (model.py): AutoModelForCausalLM + BitsAndBytesConfig
   instead of T5ForConditionalGeneration; LoRA target_modules for Llama's
   q_proj/k_proj/v_proj/o_proj instead of T5's q/v; task_type=CAUSAL_LM.

2. Prompt format (CPTDataset below): T5 tokenises phoneme input and
   target sentence as two SEPARATE sequences (encoder input / decoder
   target). A decoder-only model has no separate decoder input slot, so
   here it's a SINGLE causal sequence:
        "Phonemes: <phonemes>\\nText: <sentence><eos>"
   with the "Phonemes: ...\\nText:" prefix label-masked (-100) so the
   cross-entropy loss is only computed on the sentence being generated,
   not on the phonemes being "predicted".

3. Contrastive pooling (cpt_forward below): T5's contrastive loss pools
   model.get_encoder() hidden states. Llama has no encoder. Per the
   status summary's stated design, pooling instead happens over the
   model's OWN hidden states, restricted to the phoneme-prefix token
   span. Because attention is causal, hidden states at prefix positions
   are mathematically identical whether read off the full sequence's
   forward pass or a prefix-only forward pass (no look-ahead leakage) —
   so the anchor's prefix vector is sliced for free from the same forward
   pass used for the cross-entropy loss; the hard-negative's prefix
   vector still needs its own forward pass, exactly as the T5 version
   needed a second model.get_encoder() call for the negative.

4. Mirrors (not redesigns) one debatable detail of the original
   prototype: the "negative" companion sequence is built from the
   substituted SENTENCE TEXT in the position where phonemes normally go
   ("Phonemes: <neg_text>\\nText:"), not from a re-derived phoneme
   sequence for that substitution. This is carried over as-is from the
   Flan-T5 notebook's CPTDataset (neg_text tokenised the same way as the
   phoneme input). Worth revisiting once this moves past the dry-run
   stage — see the note in CPTDataset.__init__ below.

DRY-RUN STATUS
--------------
No Hugging Face token / Llama 3.2 access yet, and this sandbox has no
GPU (bitsandbytes 4-bit needs CUDA) — see model.py for how both of those
are handled so this same code becomes the real QLoRA run by (a) swapping
MODEL_NAME_DRYRUN -> MODEL_NAME_TARGET in model.py and (b) running on the
uni PC GPU. This script proves the pipeline mechanics: prompt formatting,
label masking, LoRA injection, phoneme-prefix pooling, contrastive loss
combination, training loop, generation, checkpoint save/load.

UPDATE (25 Jun 2026): HF gated access to meta-llama/Llama-3.2-3B was
approved. Rather than literally editing MODEL_NAME_DRYRUN in model.py
(which would also flip the default for this CPU sandbox, where there's
no GPU and no local HF auth to download a 6GB gated checkpoint with),
every CFG knob below is now also readable from an environment variable
— see the block above CFG. On the uni PC, after `huggingface-cli login`,
set CPT_MODEL_NAME=meta-llama/Llama-3.2-3B (plus CPT_N_HOMOPHONE /
CPT_N_NON_HOMOPHONE / CPT_LORA_R for the full-corpus run) and run this
same file unmodified. See RUNBOOK_real_run.md in the repo root for the
exact command sequence. Leaving CPT_* unset keeps this sandbox's CPU dry
run exactly as it was.

PHONEME SOURCE (updated 21 Jun 2026)
-------------------------------------
build_dryrun_dataframes() now pulls phonemes from
sentphonemepairs_LRS2_original.csv (the real, full-corpus LRS2 phoneme
transcriptions in original order) via
data/loader.py:load_original_phoneme_text_pairs(), instead of
self-generating them with data/g2p.py's CMU-dict G2P. A spot check found
the self-generated G2P leaves an unresolved <UNK> token on 5.95% of the
48,164 sentences (mostly British spellings like COLOUR/FLAVOUR that
aren't in the standard CMU dict), while this file has none. See that
function's docstring in loader.py for the full comparison.
"""

import os
import sys
import time
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpt_decoder.data import loader as data_loader          # noqa: E402
from cpt_decoder.augmentation.hard_negatives import generate_hard_negatives  # noqa: E402
from cpt_decoder.model import (                               # noqa: E402
    load_tokenizer, load_model_with_lora, MODEL_NAME_DRYRUN, DEVICE, USE_4BIT,
)
from cpt_decoder.evaluation.metrics import stratified_evaluate, print_results, save_results  # noqa: E402
from cpt_decoder.evaluation.error_analysis import error_category_report, print_error_report  # noqa: E402

# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION — defaults below are CPU dry-run scale. Every knob is also
# overridable via environment variable (added 25 Jun 2026) so the SAME
# script becomes the real uni-PC GPU run without editing this file:
#
#   Validation run (real model + real 4-bit, still small/fast):
#     CPT_MODEL_NAME=meta-llama/Llama-3.2-3B python3 -m src.cpt_decoder.dryrun
#
#   Full run (real model, full 37,374+10,790-row corpus, scaled LoRA):
#     CPT_MODEL_NAME=meta-llama/Llama-3.2-3B \
#     CPT_N_HOMOPHONE=37374 CPT_N_NON_HOMOPHONE=10790 \
#     CPT_LORA_R=48 CPT_EPOCHS=3 \
#     python3 -m src.cpt_decoder.dryrun
#
# Leaving every CPT_* var unset reproduces the exact CPU dry-run behaviour
# this script has always had (Qwen stand-in, 130+70 sentences) — confirmed
# unchanged by re-running it in the sandbox after adding these overrides.
# ════════════════════════════════════════════════════════════════════════
def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val else default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val else default


CFG = {
    "model_name":        _env_str("CPT_MODEL_NAME", MODEL_NAME_DRYRUN),
    # Scaled up from 20/10 (30 total) on 21 Jun 2026 for a more meaningful
    # supervisor update — still well within the 37,374 homophone / 10,790
    # non-homophone rows available in sentphonemepairs_LRS2_original.csv,
    # just a bigger stratified slice of them. 200 sentences / 2 epochs on
    # this CPU-only, 3.8GiB sandbox runs in ~45-60 min; true full-corpus
    # training is reserved for the uni GPU box (override via CPT_N_HOMOPHONE
    # / CPT_N_NON_HOMOPHONE above).
    "n_homophone":        _env_int("CPT_N_HOMOPHONE", 130),
    "n_non_homophone":    _env_int("CPT_N_NON_HOMOPHONE", 70),
    "max_input_len":      _env_int("CPT_MAX_INPUT_LEN", 96),    # phoneme-prefix budget (tokens)
    "max_target_len":     _env_int("CPT_MAX_TARGET_LEN", 32),  # sentence completion budget (tokens)
    "lora_r":             _env_int("CPT_LORA_R", 8),
    "lora_alpha":         _env_int("CPT_LORA_ALPHA", 16),
    "lora_dropout":       _env_float("CPT_LORA_DROPOUT", 0.1),
    "epochs":             _env_int("CPT_EPOCHS", 2),
    "batch_size":         _env_int("CPT_BATCH_SIZE", 2),
    "grad_accumulation":  _env_int("CPT_GRAD_ACCUM", 2),
    "learning_rate":      _env_float("CPT_LEARNING_RATE", 2e-4),
    "warmup_steps":       _env_int("CPT_WARMUP_STEPS", 2),
    "contrastive_margin": _env_float("CPT_CONTRASTIVE_MARGIN", 0.5),
    "contrastive_lambda": _env_float("CPT_CONTRASTIVE_LAMBDA", 0.1),
    "checkpoint_dir":     _env_str("CPT_CHECKPOINT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dryrun_checkpoints")),
}


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Same pooling function as the Flan-T5 prototype — architecture-agnostic."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def contrastive_loss(anchor_vec: torch.Tensor, negative_vec: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Unchanged from the Flan-T5 prototype — margin-based cosine hinge loss."""
    cos_sim = F.cosine_similarity(anchor_vec, negative_vec, dim=-1)
    return F.relu(cos_sim - margin).mean()


class CPTDataset(Dataset):
    """
    Causal-LM version of the Flan-T5 prototype's CPTDataset.

    Each item:
        input_ids / attention_mask  — "Phonemes: P\\nText: S<eos>", padded
        labels                      — same shape, -100 over the prefix + padding
        prefix_len                  — token length of "Phonemes: P\\nText:"
                                       (used to slice hidden states for pooling)
        is_homophone
        neg_input_ids / neg_attention_mask — "Phonemes: <neg_text>\\nText:"
                                       (pooling-only companion sequence; see
                                       module docstring point 4 re: this being
                                       carried over from the T5 prototype as-is)
    """

    def __init__(self, df, tokenizer, homo_set, max_input=96, max_target=32):
        self.samples = []
        max_len = max_input + max_target

        for _, row in df.iterrows():
            sentence = row["sentence"]
            phonemes = row["phonemes"]
            is_homo = sentence in homo_set

            prefix = f"Phonemes: {phonemes}\nText:"
            full_text = f"{prefix} {sentence}{tokenizer.eos_token}"

            # Use add_special_tokens=True to match how full_text is tokenized,
            # so prefix_len accounts for any BOS token prepended by the tokenizer
            # (Llama/SmolLM2 add BOS; Qwen2 does not). With add_special_tokens=False
            # the count is off by 1 on BOS-adding tokenizers: labels[:prefix_len]
            # masks one token too few, exposing the last phoneme-prefix token as a
            # training target and shifting the contrastive pooling window.
            prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
            prefix_len = min(len(prefix_ids), max_len - 1)

            enc = tokenizer(
                full_text, max_length=max_len, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            labels = input_ids.clone()
            labels[:prefix_len] = -100                       # don't score the prompt
            labels[attention_mask == 0] = -100                # don't score padding

            # Hard negative companion sequence (pooling-only; see docstring point 4)
            negs = generate_hard_negatives(sentence, max_per_word=1, max_total=1) if is_homo else []
            neg_text = negs[0]["negative"] if negs else sentence
            if not negs:
                is_homo = False

            neg_prefix = f"Phonemes: {neg_text}\nText:"
            neg_enc = tokenizer(
                neg_prefix, max_length=max_input, padding="max_length",
                truncation=True, return_tensors="pt",
            )

            self.samples.append({
                "input_ids":          input_ids,
                "attention_mask":     attention_mask,
                "labels":             labels,
                "prefix_len":         torch.tensor(prefix_len, dtype=torch.long),
                "is_homophone":       torch.tensor(is_homo, dtype=torch.bool),
                "neg_input_ids":      neg_enc["input_ids"].squeeze(0),
                "neg_attention_mask": neg_enc["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def cpt_forward(model, batch):
    """
    Causal-LM version of the Flan-T5 prototype's cpt_forward.

    Differences from the T5 version:
      - ce_loss comes from the model's own labels-masked forward pass
        (no separate decoder call needed).
      - anchor_vec is sliced from THIS SAME forward pass's hidden states,
        restricted to each example's phoneme-prefix span (causal masking
        means this is identical to a prefix-only forward pass).
      - neg_vec still needs its own forward pass (different input
        sequence), exactly as the T5 version needed model.get_encoder()
        a second time for the negative.
    """
    input_ids      = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    labels         = batch["labels"].to(DEVICE)
    prefix_len     = batch["prefix_len"].to(DEVICE)
    is_homo        = batch["is_homophone"].to(DEVICE)
    neg_input_ids  = batch["neg_input_ids"].to(DEVICE)
    neg_attn_mask  = batch["neg_attention_mask"].to(DEVICE)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True,
    )
    ce_loss = outputs.loss
    hidden_states = outputs.hidden_states[-1]   # (B, L, d_model)

    con_loss = torch.tensor(0.0, device=DEVICE)
    n_homo = is_homo.sum().item()

    if n_homo > 0:
        B, L, _ = hidden_states.shape
        # Build a per-example prefix mask (1s up to that example's prefix_len)
        positions = torch.arange(L, device=DEVICE).unsqueeze(0).expand(B, L)
        prefix_mask = (positions < prefix_len.unsqueeze(1)).float() * attention_mask.float()

        anchor_vec = mean_pool(hidden_states, prefix_mask)

        neg_outputs = model(
            input_ids=neg_input_ids,
            attention_mask=neg_attn_mask,
            output_hidden_states=True,
        )
        neg_hidden = neg_outputs.hidden_states[-1]
        neg_vec = mean_pool(neg_hidden, neg_attn_mask)

        anchor_homo = anchor_vec[is_homo]
        neg_homo = neg_vec[is_homo]
        con_loss = contrastive_loss(anchor_homo, neg_homo, margin=CFG["contrastive_margin"])

    total_loss = ce_loss + CFG["contrastive_lambda"] * con_loss
    return total_loss, ce_loss, con_loss


def build_dryrun_dataframes():
    """
    Small stratified sample using the REAL LRS2 phoneme transcriptions
    (sentphonemepairs_LRS2_original.csv via load_original_phoneme_text_pairs),
    not self-generated CMU-dict G2P — see module docstring "PHONEME SOURCE"
    note above for why.
    """
    full_df = data_loader.load_original_phoneme_text_pairs()
    homo_df, non_homo_df = data_loader.load_stratified_split(full_df)

    df_homo_sub = homo_df.head(CFG["n_homophone"]).copy()
    df_non_sub = non_homo_df.head(CFG["n_non_homophone"]).copy()
    df = pd.concat([df_homo_sub, df_non_sub], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    homo_set = set(homo_df["sentence"])

    split = max(1, int(len(df) * 0.8))
    return df[:split].reset_index(drop=True), df[split:].reset_index(drop=True), homo_set


def main():
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    print(f"Model: {CFG['model_name']}  |  Sentences: {CFG['n_homophone']} homophone + "
          f"{CFG['n_non_homophone']} non-homophone  |  LoRA r={CFG['lora_r']}  |  Epochs: {CFG['epochs']}")
    print(f"Device: {DEVICE}  |  4-bit QLoRA active: {USE_4BIT}")
    if not USE_4BIT:
        print("  -> No CUDA here, so this run validates the pipeline at full precision.")
        print("     The actual 4-bit BitsAndBytesConfig path needs the uni PC GPU.")

    print("\nLoading data (real LRS2 sentences + real phoneme transcriptions, original order)...")
    df_tr, df_val, homo_set = build_dryrun_dataframes()
    print(f"  Train: {len(df_tr)}  |  Val: {len(df_val)}")

    print(f"\nLoading tokenizer + model ({CFG['model_name']})...")
    tokenizer = load_tokenizer(CFG["model_name"])
    model = load_model_with_lora(
        CFG["model_name"], CFG["lora_r"], CFG["lora_alpha"], CFG["lora_dropout"],
        tokenizer=tokenizer,
    )

    train_ds = CPTDataset(df_tr, tokenizer, homo_set, CFG["max_input_len"], CFG["max_target_len"])
    val_ds = CPTDataset(df_val, tokenizer, homo_set, CFG["max_input_len"], CFG["max_target_len"])
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
        train_total, train_ce, train_con = 0.0, 0.0, 0.0
        optimizer.zero_grad()
        t_epoch_start = time.time()
        for step, batch in enumerate(train_dl):
            t_step_start = time.time()
            total_loss, ce, con = cpt_forward(model, batch)
            (total_loss / CFG["grad_accumulation"]).backward()
            train_total += total_loss.item()
            train_ce += ce.item()
            train_con += con.item()
            if (step + 1) % CFG["grad_accumulation"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            # Per-batch progress so a slow run is visibly progressing rather
            # than looking "stuck" (the only prior signal was one print per
            # full epoch). flush=True forces this out immediately even if
            # stdout is being piped/redirected.
            print(f"  epoch {epoch + 1} step {step + 1}/{n_tr_total}: "
                  f"loss={total_loss.item():.4f}  ({time.time() - t_step_start:.1f}s/step)",
                  flush=True)
        print(f"  -> epoch {epoch + 1} train pass took {time.time() - t_epoch_start:.1f}s total", flush=True)

        n_tr = len(train_dl)
        model.eval()
        val_total, val_ce, val_con = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_dl:
                total_loss, ce, con = cpt_forward(model, batch)
                val_total += total_loss.item()
                val_ce += ce.item()
                val_con += con.item()
        n_val = max(1, len(val_dl))

        ep_log = {
            "epoch": epoch + 1,
            "train_loss": train_total / n_tr, "train_ce": train_ce / n_tr, "train_con": train_con / n_tr,
            "val_loss": val_total / n_val, "val_ce": val_ce / n_val, "val_con": val_con / n_val,
        }
        history.append(ep_log)
        print(f"Epoch {ep_log['epoch']}: train_loss={ep_log['train_loss']:.4f} "
              f"(ce={ep_log['train_ce']:.4f}, con={ep_log['train_con']:.4f})  |  "
              f"val_loss={ep_log['val_loss']:.4f} (ce={ep_log['val_ce']:.4f}, con={ep_log['val_con']:.4f})")

    print("-" * 60)
    model.save_pretrained(CFG["checkpoint_dir"])
    tokenizer.save_pretrained(CFG["checkpoint_dir"])
    print(f"Checkpoint saved to: {CFG['checkpoint_dir']}")

    # ── Generate on the full validation split, then score it ───────────────
    print(f"\nGenerating on all {len(df_val)} validation examples...")
    model.eval()
    all_refs, all_hyps, homo_mask = [], [], []
    for _, row in df_val.iterrows():
        prompt = f"Phonemes: {row['phonemes']}\nText:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            # eos_token_id + no_repeat_ngram_size/repetition_penalty stop the
            # degenerate "the dog chased the hare..." repetition loops greedy
            # decoding is prone to on a barely-fine-tuned model — without
            # these, an unrelated rambling generation can run past the
            # reference length and inflate WER/CER past 100% on insertions
            # alone, independent of whether the content is otherwise right.
            gen = model.generate(**inputs, max_new_tokens=24, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id,
                                  repetition_penalty=1.3,
                                  no_repeat_ngram_size=3)
        decoded = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        all_refs.append(row["sentence"])
        all_hyps.append(decoded)
        homo_mask.append(row["sentence"] in homo_set)

    print("\nSample generations (first 3):")
    for ref, hyp in zip(all_refs[:3], all_hyps[:3]):
        print(f"  Ref : {ref}")
        print(f"  Gen : {hyp}")
        print()

    # WER / CER / BLEU-4 / Exact Match, overall + stratified by homophone
    # membership — this is the core dissertation metric (homophone vs
    # non-homophone performance gap), not just a loss-curve sanity check.
    eval_results = stratified_evaluate(all_refs, all_hyps, homo_mask)
    print_results(eval_results, title=f"{CFG['model_name']} dry run — generation metrics")

    metrics_csv = os.path.join(CFG["checkpoint_dir"], "metrics_log.csv")
    save_results(eval_results, metrics_csv, model_name=CFG["model_name"])

    # Error pattern analysis (P2T framework Stage 2 + Stage 3-Option-2): for
    # every substitution error, classify it as Homophone / Near-homophone /
    # Other via the CMU-dict phoneme lookups in hard_negatives.py. The
    # homophone-subset %-phonetically-explainable figure is the number that
    # tells us whether the contrastive hard-negative mechanism is targeting
    # the errors actually occurring, or whether the bottleneck is elsewhere
    # (model capacity / data scale) — see error_analysis.py module docstring.
    error_report = error_category_report(all_refs, all_hyps, homo_mask)
    print_error_report(error_report, title=f"{CFG['model_name']} dry run — error pattern analysis")

    return history


if __name__ == "__main__":
    main()
