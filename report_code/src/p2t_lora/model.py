"""P2T LoRA Decoder: Model Loading (Llama 3.2:3B + QLoRA)"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

MODEL_NAME_SMOLLM2 = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_NAME_QWEN    = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME_TARGET  = "meta-llama/Llama-3.2-3B"

MODEL_NAME_DRYRUN = MODEL_NAME_QWEN

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

CPU_DTYPE = getattr(torch, os.environ.get("CPT_CPU_DTYPE", "bfloat16"))

USE_4BIT = torch.cuda.is_available()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _select_4bit_compute_dtype() -> torch.dtype:
    """Pick bfloat16 on Ampere or newer GPUs, otherwise float16, as the 4-bit compute dtype."""
    override = os.environ.get("CPT_BNB_COMPUTE_DTYPE")
    if override:
        return getattr(torch, override)
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.bfloat16
    return torch.float16

BNB_COMPUTE_DTYPE = _select_4bit_compute_dtype()

def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer

def patch_bnb_safe_to():
    """Let accelerate's `dispatch_model` move a 4-bit bnb model. Idempotent."""
    from transformers.modeling_utils import PreTrainedModel
    if getattr(PreTrainedModel, "_bnb_safe_to_patched", False):
        return
    _orig = PreTrainedModel.to

    def _safe_to(self, *args, **kwargs):
        if (self.__class__.__name__ == "LlamaForCausalLM"
                or getattr(self, "is_loaded_in_4bit", False)
                or getattr(self, "is_loaded_in_8bit", False)):
            saved = self.__dict__.get("quantization_method", None)
            try:
                self.quantization_method = None
                return _orig(self, *args, **kwargs)
            finally:
                if saved is not None:
                    self.quantization_method = saved
        return _orig(self, *args, **kwargs)

    PreTrainedModel.to = _safe_to
    PreTrainedModel._bnb_safe_to_patched = True

def load_model_with_lora(model_name: str,
                          lora_r: int = 8,
                          lora_alpha: int = 16,
                          lora_dropout: float = 0.1,
                          tokenizer=None):
    """Load a decoder-only causal LM with QLoRA adapters."""
    quant_config = None
    if USE_4BIT:
        patch_bnb_safe_to()
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
        low_cpu_mem_usage=True,
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

    if tokenizer is not None:
        model.resize_token_embeddings(len(tokenizer))

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    compute_dtype = BNB_COMPUTE_DTYPE if USE_4BIT else CPU_DTYPE
    print(f"Loaded {model_name}  (4-bit QLoRA: {USE_4BIT}, compute dtype: {compute_dtype}, device: {DEVICE})")
    print(f"  Total parameters     : {total:>12,}")
    print(f"  Trainable (LoRA only): {trainable:>12,}  ({trainable/total*100:.2f}%)")

    return model
