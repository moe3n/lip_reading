"""
P2T LoRA Decoder — Llama 3.2:3B + QLoRA Dry Run (Phase 1 architecture port)
=========================================================================
Forked from src/cpt_decoder/ on 10 Jul 2026 (see git history for that
package's own history). cpt_decoder/ is kept exactly as it was, unmodified,
as the record of how its existing checkpoint (dryrun_checkpoints/, 65% EM)
was produced. This package is where new work happens: the phoneme-negative
fix and the contrastive-mechanism-disabled changes below, plus alignment
with a prompting-only comparison (see comparison/ and
evaluation/extended_metrics.py). Named "p2t_lora" rather than "cpt_decoder"
because CPT = Contrastive Phoneme Training, and the contrastive mechanism is
currently disabled here (see point 4) -- keeping the old name would
misdescribe what this package actually does.

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

4. CONTRASTIVE MECHANISM DISABLED (10 Jul 2026). The original prototype
   built the "negative" companion sequence from the substituted SENTENCE
   TEXT sitting in the phoneme slot ("Phonemes: <neg_text>\\nText:") --
   fixed here to re-derive real ARPAbet phonemes via data/g2p.py instead
   (stress-stripped, matching load_original_phoneme_text_pairs()'s
   cleaned "phonemes" column format). That fix then surfaced a deeper
   problem: generate_hard_negatives() sources every substitution
   candidate from get_homophones(), which returns words with the EXACT
   SAME phoneme sequence as the original by definition. So a correctly-
   re-derived negative is frequently phoneme-IDENTICAL to the anchor
   (e.g. THROUGH vs THREW) -- confirmed directly against real corpus
   sentences -- which makes "push the anchor and negative phoneme-prefix
   representations apart" an ill-posed objective for true homophones:
   there is nothing phonetically different to contrast.
   The real fix (switching the candidate source to get_near_homophones()
   -- genuinely different phoneme sequences -- for the contrastive loss
   specifically, plus caching it since it brute-force-scans the ~125k-word
   CMU dict per call) is designed but deferred: this package's first job
   is a plain (non-contrastive) LoRA SFT baseline for the prompting-vs-
   fine-tuning comparison. The mechanism is commented out below
   (CPTDataset's hard-negative block, cpt_forward's contrastive branch),
   not deleted -- see those two spots for exactly what's disabled and how
   to re-enable it.

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
import json
import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2t_lora.data import loader as data_loader          # noqa: E402
from p2t_lora.data import g2p                              # noqa: E402
from p2t_lora.augmentation.hard_negatives import generate_hard_negatives  # noqa: E402
from p2t_lora.augmentation.phoneme_noise import (            # noqa: E402
    corrupt_random, phoneme_inventory,
)
from p2t_lora.model import (                               # noqa: E402
    load_tokenizer, load_model_with_lora, MODEL_NAME_DRYRUN, DEVICE, USE_4BIT,
)
from p2t_lora.evaluation.metrics import stratified_evaluate, print_results, save_results  # noqa: E402
from p2t_lora.evaluation.error_analysis import (              # noqa: E402
    error_category_report, print_error_report, plot_error_report,
)

# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION — defaults below are CPU dry-run scale. Every knob is also
# overridable via environment variable (added 25 Jun 2026) so the SAME
# script becomes the real uni-PC GPU run without editing this file:
#
#   Validation run (real model + real 4-bit, still small/fast):
#     CPT_MODEL_NAME=meta-llama/Llama-3.2-3B python3 -m src.p2t_lora.dryrun
#
#   Full run (real model, full 37,374+10,790-row corpus, scaled LoRA):
#     CPT_MODEL_NAME=meta-llama/Llama-3.2-3B \
#     CPT_N_HOMOPHONE=37374 CPT_N_NON_HOMOPHONE=10790 \
#     CPT_LORA_R=48 CPT_EPOCHS=3 \
#     python3 -m src.p2t_lora.dryrun
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


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    return val.strip().lower() in ("1", "true", "yes", "on") if val else default


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
    # If set, overrides the two knobs above: a plain, unstratified sample of
    # this many rows straight from the full corpus, no homophone balancing.
    "n_total":            _env_int("CPT_N_TOTAL", 0),
    # If set, overrides BOTH sampling knobs above: the same fixed sequential
    # split as zero-shot/run_baseline.py and Mira's baseline — train = first
    # 45,839 rows (95.13%), val = next 1,082. The last 1,243 (test) are never
    # touched here; they stay held out for final thesis numbers. Numbers must
    # stay in sync with run_baseline.py's TRAIN_N/VAL_N/TEST_N.
    "seq_split":          _env_bool("CPT_SEQ_SPLIT", False),
    "max_input_len":      _env_int("CPT_MAX_INPUT_LEN", 96),    # phoneme-prefix budget (tokens)
    "max_target_len":     _env_int("CPT_MAX_TARGET_LEN", 32),  # sentence completion budget (tokens)
    "lora_r":             _env_int("CPT_LORA_R", 8),
    "lora_alpha":         _env_int("CPT_LORA_ALPHA", 16),
    "lora_dropout":       _env_float("CPT_LORA_DROPOUT", 0.1),
    "epochs":             _env_int("CPT_EPOCHS", 2),
    # Decoding strategy for the post-training eval generation. 1 = greedy
    # (default -- matches zero-shot/run_baseline.py and Mira's temperature-0
    # runs, so the cross-leg comparison isn't confounded by decoding).
    # 5 = beam search width 5: at each step keep the 5 most promising
    # partial sentences instead of just the single best, then return the
    # highest-scoring finished one. Roughly N times slower generation.
    # Decoding is post-training -- this can also be changed and re-run on a
    # saved checkpoint without retraining anything.
    "num_beams":          _env_int("CPT_NUM_BEAMS", 1),
    # Noise-augmented training (added 17 Jul 2026). Corrupts the phoneme INPUT
    # of training examples only -- validation is never corrupted, so every
    # number stays comparable to the clean runs. noise_prob=0 (default) is the
    # existing clean-training behaviour, unchanged.
    #   noise_prob      fraction of training examples that get corrupted
    #   noise_rate_min/max  corruption rate range, sampled per example
    # See augmentation/phoneme_noise.py for what the corruptions are and why.
    "noise_prob":         _env_float("CPT_NOISE_PROB", 0.0),
    "noise_rate_min":     _env_float("CPT_NOISE_RATE_MIN", 0.05),
    "noise_rate_max":     _env_float("CPT_NOISE_RATE_MAX", 0.15),
    "noise_seed":         _env_int("CPT_NOISE_SEED", 42),
    # Global training seed (added 17 Jul 2026). Fixes DataLoader shuffle order
    # and LoRA initialisation so a run is exactly reproducible, not merely
    # stable. The two earlier clean runs landed within 0.0002 train loss of
    # each other unseeded, so this changes reproducibility, not results.
    "seed":               _env_int("CPT_SEED", 42),
    "batch_size":         _env_int("CPT_BATCH_SIZE", 2),
    "grad_accumulation":  _env_int("CPT_GRAD_ACCUM", 2),
    "learning_rate":      _env_float("CPT_LEARNING_RATE", 2e-4),
    "warmup_steps":       _env_int("CPT_WARMUP_STEPS", 2),
    "contrastive_margin": _env_float("CPT_CONTRASTIVE_MARGIN", 0.5),
    "contrastive_lambda": _env_float("CPT_CONTRASTIVE_LAMBDA", 0.1),
    # Own default dir (not cpt_decoder's dryrun_checkpoints/) so a default-
    # settings run here can never overwrite the old package's existing
    # checkpoint -- see module docstring.
    "checkpoint_dir":     _env_str("CPT_CHECKPOINT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "p2t_lora_checkpoints")),
    # Stage 3 Option 5 (LLM-based error classification) escalation gate --
    # added 25 Jun 2026, see evaluation/error_analysis.py /
    # evaluation/llm_judge.py. Off by default: Option 5 needs a
    # disable_adapter() pass through the already-loaded model below for
    # every Homophone/Near-homophone substitution Option 3 (grammar)
    # leaves unresolved, which is a real cost increase on top of the
    # generation loop already happening, and -- per llm_judge.py's own
    # documented finding -- the CPU dry-run judge model's classifications
    # aren't yet trustworthy even when it runs cleanly. Set
    # CPT_LLM_ERROR_JUDGE=1 to opt in (e.g. once a genuinely instruct-
    # tuned judge model is available on the uni PC GPU -- see
    # llm_judge.py's docstring on why disable_adapter() alone doesn't fix
    # this for MODEL_NAME_TARGET, a base/non-instruct checkpoint).
    "llm_error_judge":    _env_bool("CPT_LLM_ERROR_JUDGE", False),
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
        is_homophone                — kept for stratified eval/reporting even
                                       though the contrastive loss that used
                                       to consume it is disabled (module
                                       docstring point 4)
        neg_input_ids / neg_attention_mask — DISABLED, see __init__ below.
    """

    def __init__(self, df, tokenizer, homo_set, max_input=96, max_target=32,
                 noise=None):
        """
        noise: None (clean, the default) or a dict with keys prob / rate_min /
        rate_max / seed / inventory. Passed for the TRAINING split only -- the
        validation split must stay clean or the metrics stop being comparable
        to every earlier run.

        ponytail: corruption is applied once here, at dataset build time, so a
        given example keeps the same noise across all epochs. Re-sampling per
        epoch would be stronger augmentation but needs tokenisation moved into
        __getitem__; worth doing only if 3 epochs of fixed noise proves too weak.
        """
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

            # ── Hard negative companion sequence — DISABLED (module docstring
            # point 4: exact-homophone negatives are frequently phoneme-
            # identical to the anchor, an ill-posed contrastive target; the
            # near-homophone fix is designed but deferred). Commented out,
            # not deleted, so re-enabling is a one-block uncomment once that
            # fix lands.
            # negs = generate_hard_negatives(sentence, max_per_word=1, max_total=1) if is_homo else []
            # neg_sentence = negs[0]["negative"] if negs else sentence
            # neg_phonemes = " ".join(g2p.sentence_to_phoneme_list(neg_sentence, stress=False))
            # neg_prefix = f"Phonemes: {neg_phonemes}\nText:"
            # neg_enc = tokenizer(
            #     neg_prefix, max_length=max_input, padding="max_length",
            #     truncation=True, return_tensors="pt",
            # )

            self.samples.append({
                "input_ids":          input_ids,
                "attention_mask":     attention_mask,
                "labels":             labels,
                "prefix_len":         torch.tensor(prefix_len, dtype=torch.long),
                "is_homophone":       torch.tensor(is_homo, dtype=torch.bool),
                # "neg_input_ids":      neg_enc["input_ids"].squeeze(0),
                # "neg_attention_mask": neg_enc["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def cpt_forward(model, batch):
    """
    Causal-LM version of the Flan-T5 prototype's cpt_forward.

    Contrastive branch DISABLED (module docstring point 4) — this now just
    runs the masked-label cross-entropy forward pass. con_loss is still
    returned as a zero tensor so the training loop's logging doesn't need
    its own special-casing. output_hidden_states is no longer requested
    either, since the only consumer (contrastive pooling) is off — that
    also lightens the forward pass a bit while this is disabled.
    """
    input_ids      = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    labels         = batch["labels"].to(DEVICE)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    ce_loss = outputs.loss
    con_loss = torch.tensor(0.0, device=DEVICE)

    # ── Contrastive loss — DISABLED, see module docstring point 4 ──────────
    # prefix_len     = batch["prefix_len"].to(DEVICE)
    # is_homo        = batch["is_homophone"].to(DEVICE)
    # neg_input_ids  = batch["neg_input_ids"].to(DEVICE)
    # neg_attn_mask  = batch["neg_attention_mask"].to(DEVICE)
    # hidden_states  = outputs.hidden_states[-1]   # (B, L, d_model) -- needs output_hidden_states=True above
    # n_homo = is_homo.sum().item()
    # if n_homo > 0:
    #     B, L, _ = hidden_states.shape
    #     positions = torch.arange(L, device=DEVICE).unsqueeze(0).expand(B, L)
    #     prefix_mask = (positions < prefix_len.unsqueeze(1)).float() * attention_mask.float()
    #     anchor_vec = mean_pool(hidden_states, prefix_mask)
    #     neg_outputs = model(
    #         input_ids=neg_input_ids,
    #         attention_mask=neg_attn_mask,
    #         output_hidden_states=True,
    #     )
    #     neg_hidden = neg_outputs.hidden_states[-1]
    #     neg_vec = mean_pool(neg_hidden, neg_attn_mask)
    #     anchor_homo = anchor_vec[is_homo]
    #     neg_homo = neg_vec[is_homo]
    #     con_loss = contrastive_loss(anchor_homo, neg_homo, margin=CFG["contrastive_margin"])

    total_loss = ce_loss  # + CFG["contrastive_lambda"] * con_loss  -- disabled
    return total_loss, ce_loss, con_loss


def build_dryrun_dataframes():
    """
    Uses the REAL LRS2 phoneme transcriptions (sentphonemepairs_LRS2_original.csv
    via load_original_phoneme_text_pairs), not self-generated CMU-dict G2P —
    see module docstring "PHONEME SOURCE" note above for why.

    Training-set selection: CPT_N_TOTAL set -> a plain unstratified sample of
    that many rows straight from the full corpus. Unset (default) -> the
    original stratified approach, n_homophone + n_non_homophone rows pulled
    separately then combined, so homophone-containing sentences are
    deliberately over-represented relative to their share of the corpus.
    Either way, eval is still stratified by homophone membership afterwards
    (homo_set below) — that's a reporting split, not a training-data choice,
    and stays on regardless.
    """
    full_df = data_loader.load_original_phoneme_text_pairs()
    homo_df, non_homo_df = data_loader.load_stratified_split(full_df)
    homo_set = set(homo_df["sentence"])

    if CFG["seq_split"]:
        TRAIN_N, VAL_N = 45839, 1082   # matches zero-shot/run_baseline.py; test rows stay untouched
        train_df = full_df.iloc[:TRAIN_N].reset_index(drop=True)
        val_df   = full_df.iloc[TRAIN_N:TRAIN_N + VAL_N].reset_index(drop=True)
        # Remove val sentences that appear verbatim in training (LRS2 has repeated
        # broadcast phrases across the corpus). Training rows are unchanged.
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

    # Noise augmentation applies to training only. df_val is built clean so the
    # reported metrics stay directly comparable to the clean-training runs.
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

    train_ds = CPTDataset(df_tr, tokenizer, homo_set, CFG["max_input_len"],
                          CFG["max_target_len"], noise=noise_cfg)
    val_ds = CPTDataset(df_val, tokenizer, homo_set, CFG["max_input_len"], CFG["max_target_len"])
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

        # Crash insurance AND overfitting insurance for multi-hour runs: save
        # each epoch's adapter to its own subdir (epoch_1/, epoch_2/, ...)
        # rather than overwriting. If the per-epoch val_loss in
        # training_history.json shows a later epoch got WORSE (overfitting),
        # the earlier, better adapter still exists -- generation can be
        # re-run from it without retraining. ~1.6GB per epoch on disk (most
        # of it the resized-vocab embedding table PEFT insists on saving).
        model.save_pretrained(os.path.join(CFG["checkpoint_dir"], f"epoch_{epoch + 1}"))
        with open(os.path.join(CFG["checkpoint_dir"], "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"  (epoch {epoch + 1} adapter saved to epoch_{epoch + 1}/)", flush=True)

    print("-" * 60)
    model.save_pretrained(CFG["checkpoint_dir"])
    tokenizer.save_pretrained(CFG["checkpoint_dir"])
    print(f"Checkpoint saved to: {CFG['checkpoint_dir']}")

    # ── Generate on the full validation split, then score it ───────────────
    print(f"\nGenerating on all {len(df_val)} validation examples...")
    model.eval()
    all_phonemes, all_refs, all_hyps, homo_mask = [], [], [], []
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
            # max_new_tokens=34: measured against the sequential val split's
            # actual sentences -- the longest needs 28 tokens (+eos), and 8 of
            # 1,082 (0.7%) need more than the previous cap of 24, which was
            # cutting them off mid-sentence (34/1000 predictions in the
            # 5,000-row run sat at the cap). 34 covers every sentence in the
            # corpus with margin, and matches zero-shot/run_baseline.py's
            # budget. Costs nothing on normal rows: generation stops at the
            # learned <eos> long before the cap.
            gen_kwargs = dict(max_new_tokens=34, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id,
                              eos_token_id=tokenizer.eos_token_id,
                              repetition_penalty=1.3,
                              no_repeat_ngram_size=3)
            if CFG["num_beams"] > 1:
                gen_kwargs.update(num_beams=CFG["num_beams"], early_stopping=True)
            gen = model.generate(**inputs, **gen_kwargs)
        decoded = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        # Keep only the first line, same extraction rule as zero-shot's
        # extract_answer(): an under-trained model that hasn't learned the
        # eos signal rambles onto new lines after its answer (e.g. inventing
        # a "Phonetic Transcription: ..." continuation), and that ramble
        # shouldn't contaminate the score. References are all single-line,
        # so this can never clip a correct answer.
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
    # tells us whether homophone confusions are phonetically explainable, or
    # whether the bottleneck is elsewhere (model capacity / data scale) —
    # see error_analysis.py module docstring.
    error_report = error_category_report(
        all_refs, all_hyps, homo_mask,
        tokenizer=tokenizer, model=model, use_llm=CFG["llm_error_judge"],
    )
    print_error_report(error_report, title=f"{CFG['model_name']} dry run — error pattern analysis")

    error_report_path = os.path.join(CFG["checkpoint_dir"], "error_report.json")
    with open(error_report_path, "w", encoding="utf-8") as f:
        json.dump(error_report, f, indent=2, ensure_ascii=False, default=str)

    error_chart_path = os.path.join(CFG["checkpoint_dir"], "error_report.png")
    plot_error_report(error_report, error_chart_path,
                       title=f"{CFG['model_name']} — substitution error categories")
    print(f"Detailed error pattern analysis saved to: {error_report_path}")
    print(f"Error pattern chart saved to: {error_chart_path}")

    return history


if __name__ == "__main__":
    main()
