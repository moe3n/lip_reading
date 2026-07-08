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

---

## Stage 1b — zero-shot baseline (Llama 3.2:3B, no training)

Pure zero-shot evaluation of the real Llama 3.2:3B base model on the
48,164-row LRS2 phoneme→text task. Deterministic 95.13 / 2.26 / 2.59%
sequential split:

- **train**: first 45,839 rows
- **val**: next 1,082 rows
- **test**: last 1,243 rows

Output goes to `zero-shot/llama3.2_3b/` (parallel to the existing
`zero-shot/other's work/`) and the JSON layout
(`{results: [...], total_samples: N}` per sample with `index,
input_phonemes, expected_text, model_output, time_seconds`) is **byte-
comparable** to `zero-shot/other's work/ZeroShot_baseline_results_
45840_samples.json` so `diff` on `input_phonemes`/`expected_text`
arrays should be empty.

Per-split invocation (recommended — the train split is the slow one,
run it overnight):

```powershell
cd C:\Projects\lip_reading
.venv\Scripts\activate

# Val first (smallest, sanity-check the pipeline is correct)
$env:CPT_ZS_MODEL_NAME = "meta-llama/Llama-3.2-3B"
$env:CPT_ZS_SPLIT      = "val"
python -m src.cpt_decoder.zero_shot.run

# Then test
$env:CPT_ZS_SPLIT = "test"
python -m src.cpt_decoder.zero_shot.run

# Finally train (overnight -- ~45k samples)
$env:CPT_ZS_SPLIT = "train"
python -m src.cpt_decoder.zero_shot.run
```

To run all three splits in one shot (don't launch this without first
confirming the val output looks sane):

```powershell
$env:CPT_ZS_MODEL_NAME = "meta-llama/Llama-3.2-3B"
$env:CPT_ZS_SPLIT      = "all"
python -m src.cpt_decoder.zero_shot.run
```

Configurable env vars:

- `CPT_ZS_MODEL_NAME` — default `meta-llama/Llama-3.2-3B`. Use
  `unsloth/Llama-3.2-3B` (ungated tokenizer-identical mirror) if HF
  gated access still isn't available; weights still download from HF.
- `CPT_ZS_SPLIT` — `train` / `val` / `test` / `all`. Default `all`.
- `CPT_ZS_OUTPUT_DIR` — default `zero-shot/llama3.2_3b/`.
- `CPT_ZS_BATCH_SIZE` — default `8` on CUDA, `1` on CPU.
- `CPT_ZS_MAX_NEW_TOK` — default `34` (matches
  `verify_token_budget.TARGET_BUDGET`, covers ~99% of LRS2 sentences
  without truncation).

**Resume support.** Each split streams results to a
`zeroshot_<split>_<N>.jsonl` side-car one record per line. Killing the
process and restarting with the same env vars picks up at the next
unwritten row — no re-decoding. Useful for the multi-hour train split.

**CPU fallback** is built in for pipeline-smoke-testing (loads the
`Qwen/Qwen2.5-0.5B-Instruct` stand-in in bf16, no 4-bit). It prints
`4-bit QLoRA active: False` and a notice that the numbers are not
thesis results. Don't mistake it for a real run.

**What NOT to do.** Don't set `repetition_penalty` or
`no_repeat_ngram_size` here — the prior baseline ran plain greedy. Using
different decoding settings on the new model would break the
apples-to-apples comparison the thesis is built around. The penalties
in `dryrun.py` are tuned for a *trained* decoder and would confound a
zero-shot number.

After all three splits finish, the cross-comparison checklist:

1. `cd zero-shot` then
   `(Get-Content .\other's work\ZeroShot_baseline_results_45840_samples.json | ConvertFrom-Json).results[0..5] | %{ $_.input_phonemes }`
   must match the first 6 `input_phonemes` of
   `llama3.2_3b\zeroshot_train_45839.json` byte-for-byte (corpus and
   row order are fixed).
2. Two consecutive runs of the val split with `do_sample=False` must
   produce byte-identical `model_output` arrays. If they don't, a
   non-determinism source leaked in (e.g. an unintended RNG seeding
   path) and the thesis numbers aren't trustworthy yet.
3. `metrics_summary.csv` should have 9 rows (3 splits ×
   overall/homophone/non-homophone). `print_results()` per split is
   the live log.

Stage 1b validates **number**, not pipeline. The trained-decoder
comparison still lives in `dryrun.py` (Stages 1 + 2 above).
