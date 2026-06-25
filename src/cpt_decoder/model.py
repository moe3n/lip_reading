"""
CPT Decoder — Model Loading (Llama 3.2:3B + QLoRA)
=====================================================
Replaces the Flan-T5 (encoder-decoder) loading code in
CPT_Decoder_Phase2_Mini.ipynb with a decoder-only causal LM + QLoRA setup,
per the dissertation's stated approach (Project_Status_Summary.md, Llama
3.2:3B + QLoRA via bitsandbytes + LoRA/PEFT, chosen to match VALLR).

WHY THIS FILE LOOKS DIFFERENT FROM THE FLAN-T5 PROTOTYPE:
  - T5ForConditionalGeneration -> AutoModelForCausalLM. Llama has no
    separate encoder, so there is no model.get_encoder() — see
    cpt_forward() in train.py for how contrastive pooling is redone.
  - LoRA target_modules change from T5's ["q", "v"] to Llama's attention
    projection names: q_proj, k_proj, v_proj, o_proj. task_type changes
    from SEQ_2_SEQ_LM to CAUSAL_LM.
  - 4-bit quantisation (BitsAndBytesConfig) is the "Q" in QLoRA. It is a
    CUDA-only feature of bitsandbytes — there is no CPU/MPS backend for
    the NF4 training kernels as of this writing. USE_4BIT below auto-
    detects this so the same code runs (unquantised) on a CPU/Mac dry
    run and (quantised) on the uni PC GPU, without edits.

DRY-RUN STATUS: MODEL_NAME_DRYRUN points at a small, non-gated stand-in
because Hugging Face access + token for the gated meta-llama/Llama-3.2-3B
checkpoint are not set up yet. Two stand-ins are kept below:
  - MODEL_NAME_SMOLLM2 (135M) — the safest fallback; confirmed working
    in this sandbox (3.8GiB RAM, no GPU).
  - MODEL_NAME_QWEN (Qwen2.5-0.5B-Instruct) — requested for sharing
    results with the supervisor (closer in scale/quality to a real
    run than the 135M stand-in). Two attempts at fp32 on the CPU path
    (with and without low_cpu_mem_usage=True) were silently OOM-killed
    here (exit 137) right after weight loading completed — fp32 weights
    alone are ~2GB on a box with ~3.4GiB available. Switching the CPU
    load dtype to bfloat16 (see load_model_with_lora() below) halved the
    resident weight footprint to ~1GB and fixed it: confirmed working
    end-to-end (21 Jun 2026) — load, 2-epoch LoRA fine-tune, checkpoint
    save, generation smoke test, all completed without a memory error.
Both use Llama-style module names (q_proj/k_proj/v_proj/o_proj/gate_proj/
up_proj/down_proj), so LORA_TARGET_MODULES below transfers to Llama
3.2:3B unchanged — swapping MODEL_NAME_DRYRUN -> MODEL_NAME_TARGET is
meant to be the only change needed once HF access is granted and this
runs on the uni PC GPU.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

# ── Swap this when HF access to meta-llama/Llama-3.2-3B is granted ───────────
MODEL_NAME_SMOLLM2 = "HuggingFaceTB/SmolLM2-135M-Instruct"  # non-gated, Llama-arch stand-in (safest)
MODEL_NAME_QWEN    = "Qwen/Qwen2.5-0.5B-Instruct"            # non-gated, Llama-arch stand-in (larger)
MODEL_NAME_TARGET  = "meta-llama/Llama-3.2-3B"               # real target (gated; needs HF token)

MODEL_NAME_DRYRUN = MODEL_NAME_QWEN

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# CPU-path load dtype for the base model. Defaults to bfloat16 — this was
# chosen to fix an OOM kill on a very RAM-constrained sandbox (3.8GiB total)
# where fp32 Qwen2.5-0.5B weights alone (~2GB) left too little headroom.
# On a machine with more RAM (e.g. a laptop with 8GB+), fp32 may actually
# run *faster*, since not every CPU has accelerated bf16 GEMM kernels in
# PyTorch and some ops silently upcast to fp32 per-call anyway, adding
# overhead without saving compute time (only memory). Override via:
#   CPT_CPU_DTYPE=float32 python3 dryrun.py
CPU_DTYPE = getattr(torch, os.environ.get("CPT_CPU_DTYPE", "bfloat16"))

USE_4BIT = torch.cuda.is_available()   # bitsandbytes 4-bit needs CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_4bit_compute_dtype() -> torch.dtype:
    """
    Added 25 Jun 2026, ahead of the first real run on the uni PC's GPUs.

    bfloat16 has no hardware support below compute capability 8.0
    (Ampere+) — PyTorch/bitsandbytes will raise "ValueError: Bfloat16 is
    only supported on GPUs with compute capability of at least 8.0" the
    moment training starts if it's requested on an older card. The uni
    PC's GTX 1080s are Pascal, compute capability 6.1, so the previous
    hardcoded bnb_4bit_compute_dtype=torch.bfloat16 below would have hit
    this immediately. Auto-detect instead of hardcoding to one known GPU,
    so this keeps working if the hardware ever changes. Override with
    CPT_BNB_COMPUTE_DTYPE=bfloat16|float16 if you ever need to force it.
    """
    override = os.environ.get("CPT_BNB_COMPUTE_DTYPE")
    if override:
        return getattr(torch, override)
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.bfloat16
    return torch.float16


BNB_COMPUTE_DTYPE = _select_4bit_compute_dtype()  # bf16 on Ampere+, fp16 on Pascal/Volta/Turing


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Add a dedicated pad token rather than aliasing EOS. When pad == EOS,
    # generate() treats the first padding token as a stop signal (empty output
    # = 100% WER for that sample) and training sees spurious EOS signals in
    # every padded position. A separate [PAD] token avoids both problems.
    # resize_token_embeddings() must be called on the model after this.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def load_model_with_lora(model_name: str,
                          lora_r: int = 8,
                          lora_alpha: int = 16,
                          lora_dropout: float = 0.1,
                          tokenizer=None):
    """
    Load a decoder-only causal LM with QLoRA adapters.

    On CUDA: loads in 4-bit (NF4, double-quant, bf16 compute) — the
             actual QLoRA path described in the proposal.
    On CPU/MPS: loads at full precision with the same LoRA config, so
             the rest of the pipeline (data flow, loss, training loop)
             can still be exercised end-to-end. The 4-bit path itself
             can only be verified on CUDA hardware (uni PC GPU).
    """
    quant_config = None
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=BNB_COMPUTE_DTYPE,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=BNB_COMPUTE_DTYPE if USE_4BIT else CPU_DTYPE,
        device_map="auto" if USE_4BIT else None,
        low_cpu_mem_usage=True,   # avoid a transient ~2x RAM spike during from_pretrained()
    )
    if not USE_4BIT:
        base_model = base_model.to(DEVICE)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)

    # Resize to account for any new special tokens added by load_tokenizer
    # (e.g. the dedicated [PAD] token). Safe no-op when vocab size is unchanged.
    if tokenizer is not None:
        model.resize_token_embeddings(len(tokenizer))

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    compute_dtype = BNB_COMPUTE_DTYPE if USE_4BIT else CPU_DTYPE
    print(f"Loaded {model_name}  (4-bit QLoRA: {USE_4BIT}, compute dtype: {compute_dtype}, device: {DEVICE})")
    print(f"  Total parameters     : {total:>12,}")
    print(f"  Trainable (LoRA only): {trainable:>12,}  ({trainable/total*100:.2f}%)")

    return model
