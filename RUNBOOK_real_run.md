# Real Run Runbook — Llama 3.2:3B + QLoRA on the Uni PC

Written 25 Jun 2026, once HF gated access to `meta-llama/Llama-3.2-3B` was
approved. This is the exact command sequence to go from "dry run on a CPU
stand-in" to "real model, true 4-bit QLoRA, on the uni PC's GPU." Every
step below is meant to be run **on the uni PC itself**, in a PowerShell /
WSL terminal — nothing here runs from this assistant's sandbox.

No Hugging Face token should ever be pasted into a chat with Claude or
anyone else. Steps 3 and 4 below are things you run locally; the token
itself never needs to leave that terminal.

---

## 0. What changed in the code, and why

Three files were edited to make this possible. They are **not committed
yet** — see "What to do with these changes" at the bottom. Summary of
what changed:

- **`src/cpt_decoder/model.py`** — the 4-bit compute dtype was hardcoded
  to `torch.bfloat16`. The uni PC's GTX 1080s are Pascal (compute
  capability 6.1); bfloat16 needs compute capability ≥ 8.0 (Ampere+), so
  the old code would have failed immediately on this hardware with
  `ValueError: Bfloat16 is only supported on GPUs with compute capability
  of at least 8.0`. It's now auto-detected (`_select_4bit_compute_dtype()`)
  — falls back to `float16` on the 1080s, would use `bfloat16` automatically
  if this code ever runs on newer GPUs.
- **`src/cpt_decoder/data/loader.py`** — `clean_phoneme_seq()` now strips
  the literal `"<space>"` marker instead of leaving it in. Verified via
  `check_token_lengths.py` against the full 48,164-row corpus: this drops
  the truncation rate at `max_input_len=96` from 7.81% to **0.00%**
  (better than the originally-estimated 0.31% — re-running the audit
  after the fix confirmed it).
- **`src/cpt_decoder/dryrun.py`** — every value in `CFG` is now also
  readable from an environment variable (`CPT_MODEL_NAME`,
  `CPT_N_HOMOPHONE`, `CPT_LORA_R`, etc.). Leaving them all unset reproduces
  the exact CPU dry-run behaviour this script always had — confirmed with
  a smoke test after editing. This means the **same script**, unmodified,
  becomes the real run just by setting environment variables before
  calling it — no more hand-editing `MODEL_NAME_DRYRUN` in `model.py`.
- **`test_gpu.py`** — now also prints each GPU's compute capability and a
  one-line note on whether bf16 will be used, so you can confirm the
  auto-detect picked the right dtype before committing to a long run.

## 1. Pull the latest code

```powershell
cd C:\Projects\lip_reading
git pull
```

(If you're working from this assistant's edits directly rather than a
push from elsewhere, make sure these three files — `model.py`,
`loader.py`, `dryrun.py` — and `test_gpu.py` / this runbook actually
reach the uni PC's checkout before continuing. `git status` should show
no pending changes once they're in.)

## 2. Add the CPT Decoder's extra dependencies

The LEAP setup guide's venv has torch/transformers/etc. but not the QLoRA
stack. From the activated venv:

```powershell
.venv\Scripts\activate
pip install -r requirements-cpt-decoder.txt
```

This adds `peft`, `bitsandbytes`, `accelerate` (the QLoRA stack),
`nltk`, `jiwer`, `sacrebleu` (data + eval, reused from Phase 1).

## 3. Confirm the GPUs and dtype auto-detect

```powershell
python test_gpu.py
```

Expect `CUDA available: True`, `GPU count: 2`, and a compute capability
of `6.1` for both GTX 1080s, with the note confirming float16 will be
used. If compute capability shows `8.0+` instead (e.g. you're actually on
different hardware than the setup guide describes), the auto-detect will
pick bf16 instead — no code change needed either way.

## 4. Authenticate with Hugging Face (local only — no token in chat)

You said the gated-access request for `meta-llama/Llama-3.2-3B` is now
**accepted** but local auth isn't configured yet. Generate a token at
https://huggingface.co/settings/tokens (read access is enough) and run,
in the same terminal:

```powershell
huggingface-cli login
```

Paste the token when prompted — this stays local to that terminal/config
file and is never something to share in chat. Confirm it worked:

```powershell
python -c "from huggingface_hub import whoami; print(whoami())"
```

## 5. Stage 1 — validation run (real model, real 4-bit, small scale)

Before committing GPU time to the full corpus, confirm the real model +
true 4-bit path actually works end-to-end. This uses the same 130+70
sentence scale as every CPU dry run so far — just swapping in the real
model and the real GPU:

```powershell
cd C:\Projects\lip_reading
.venv\Scripts\activate
$env:CPT_MODEL_NAME = "meta-llama/Llama-3.2-3B"
python -m src.cpt_decoder.dryrun
```

What to check in the output:

- `4-bit QLoRA active: True` (confirms bitsandbytes engaged, not a silent
  CPU fallback)
- `compute dtype: torch.float16` (confirms the Pascal auto-detect worked
  — if this prints `torch.bfloat16` instead, something's off; check
  `test_gpu.py`'s compute-capability reading)
- It completes all epochs without a CUDA error, saves a checkpoint, and
  prints generation metrics + the error-pattern report

This run downloads the actual `meta-llama/Llama-3.2-3B` checkpoint the
first time (full-precision safetensors, ~6GB — do this on uni network,
same as the LEAP guide's note about the Ollama 8B pull). It then gets
quantized to 4-bit at load time, so resident VRAM is only ~1.5–2GB —
either 1080 alone is plenty; you don't need both.

If this stage fails, that's exactly the point of running it small first
— a 5-minute failure here is much cheaper than discovering the same
problem after committing to the full corpus.

## 6. Stage 2 — full run (full corpus, scaled LoRA)

Once Stage 1 looks right, scale up. `Project_Status_Summary.md` calls out
`lora_r=48` as the original prototype's note for the full-corpus run;
adjust `CPT_EPOCHS` based on how Stage 1's per-step timing extrapolates
(the script already prints `(Xs/step)` per step — use that to estimate
total time before launching a multi-hour job):

```powershell
$env:CPT_MODEL_NAME = "meta-llama/Llama-3.2-3B"
$env:CPT_N_HOMOPHONE = "37374"
$env:CPT_N_NON_HOMOPHONE = "10790"
$env:CPT_LORA_R = "48"
$env:CPT_EPOCHS = "3"        # adjust after seeing Stage 1's s/step
python -m src.cpt_decoder.dryrun
```

Unset the `$env:CPT_*` variables afterward (or just close the terminal)
so a future CPU dry run on the Mac doesn't accidentally inherit them.

## 7. If something goes wrong

- **`ValueError: Bfloat16 is only supported...`** — shouldn't happen after
  this fix, but if it does, it means `_select_4bit_compute_dtype()` in
  `model.py` mis-detected the GPU. Override directly:
  `$env:CPT_BNB_COMPUTE_DTYPE = "float16"`.
- **Out of memory** — unexpected at 3B/4-bit on an 8GB card, but if it
  happens, set `$env:CUDA_VISIBLE_DEVICES = "0"` to force single-GPU and
  rule out `device_map="auto"` splitting the model awkwardly across both
  1080s over PCIe.
- **Gated-repo 403 / "you need to request access"** — re-run step 4's
  `whoami()` check; if it doesn't show your account, the local login
  didn't persist.

## What to do with these changes

These three files are edited locally but **not committed**. Review the
diff (`git diff`), and when it looks right, commit and push from
wherever you're comfortable doing that — this runbook intentionally
doesn't do it for you.
