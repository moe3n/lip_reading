"""
Verify accelerate works correctly on Windows in this venv.
Tests progressively: import -> config -> device placement -> training step.
"""

import sys
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print()

# ── 1. Import check ───────────────────────────────────────────────────────────
print("1. Import check")
try:
    import accelerate
    print(f"   accelerate version: {accelerate.__version__}")
except ImportError as e:
    print(f"   FAIL: {e}")
    sys.exit(1)

# ── 2. Accelerator initialization ─────────────────────────────────────────────
print("\n2. Accelerator initialization")
try:
    from accelerate import Accelerator
    accel = Accelerator()
    print(f"   device:           {accel.device}")
    print(f"   distributed type: {accel.distributed_type}")
    print(f"   num processes:    {accel.num_processes}")
    print(f"   mixed precision:  {accel.mixed_precision}")
except Exception as e:
    print(f"   FAIL: {e}")
    sys.exit(1)

# ── 3. Device placement of a tiny model ───────────────────────────────────────
print("\n3. Device placement test")
try:
    import torch
    import torch.nn as nn

    model = nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model, optimizer = accel.prepare(model, optimizer)
    print(f"   model device: {next(model.parameters()).device}")
    print(f"   model dtype:  {next(model.parameters()).dtype}")
except Exception as e:
    print(f"   FAIL: {e}")
    sys.exit(1)

# ── 4. One forward-backward step ──────────────────────────────────────────────
print("\n4. Forward-backward step with accel.backward()")
try:
    x = torch.randn(4, 10).to(accel.device)
    y = torch.randint(0, 2, (4,)).to(accel.device)

    optimizer.zero_grad()
    out = model(x)
    loss = nn.functional.cross_entropy(out, y)
    accel.backward(loss)
    optimizer.step()
    print(f"   loss: {loss.item():.4f}")
    print(f"   step completed without errors")
except Exception as e:
    print(f"   FAIL: {e}")
    sys.exit(1)

# ── 5. Integration with transformers + bitsandbytes (the real test) ──────────
print("\n5. Loading a small quantised model via accelerate device_map")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    small_model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        quantization_config=bnb,
        device_map="auto",  # this is the accelerate hook
    )
    print(f"   model placed on: {small_model.device}")
    print(f"   device_map handed off to accelerate: OK")
except Exception as e:
    print(f"   FAIL: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All accelerate checks passed. Safe to use in the pipeline.")
print("=" * 60)