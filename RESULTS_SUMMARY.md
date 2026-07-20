# P2T Decoder — Results Summary (for supervisor)

> **Current as of 19 July 2026.** See also:
> - `p2t_lora_checkpoints_dedup/NOTES.md` — working notes for the staged error analysis.
> - `meeting_2026-07-08/Progress_Briefing.md` — dated snapshot of the 8 July 2026 supervisor briefing (superseded by this file).

**Model:** `meta-llama/Llama-3.2-3B` (decoder-only) fine-tuned with **QLoRA**
(4-bit NF4 base + LoRA adapters).
**Task:** Phoneme → English sentence decoding (P2T) on the **LRS2** corpus, with
contrastive hard-negative training aimed at homophone disambiguation.
**Run:** dedup run, **`p2t_lora_checkpoints_dedup/`** (epoch_3 adapter), decoded with **beam search, width = 5**.

## Dataset

| Split | Sentences | Notes |
|---|---|---|
| Full corpus | 48,164 | LRS2 sentence–phoneme pairs (`sentphonemepairs_LRS2_original.csv`) |
| Train | 45,839 | sequential split, rows 1–45,839 |
| Val (raw) | 1,082 | sequential split, rows 45,840–46,921 |
| **Val (dedup, headline eval)** | **949** | 133 sentences removed (appear verbatim in train — LRS2 repeats broadcast phrases) |
| Test | 1,243 | sequential split, rows 46,922–48,164 — **never touched, stays held out** |
| — Homophone subset (val dedup) | 672 | sentences containing ≥1 homophone-prone word |
| — Non-homophone subset (val dedup) | 277 | |

The split is **sequential** to match the official LRS2 split sizes (matches LRS2's own convention rather than random). Dedup is applied **in memory** inside `build_dryrun_dataframes()`; the CSV is never modified. Training rows are never filtered.

## Headline results (dedup val, n = 949, beam-5)

| Subset | n | WER ↓ | CER ↓ | BLEU-4 ↑ | Exact Match ↑ |
|---|---|---|---|---|---|
| **Overall** | **949** | **2.09%** | **0.98%** | **0.9673** | **91.99%** |
| Homophone | 672 | 1.83% | 0.85% | 0.9686 | 92.56% |
| Non-homophone | 277 | 2.95% | 1.35% | 0.9622 | 90.61% |

**Headline:** the fine-tuned model decodes phonemes to text very strongly — **2.1% WER, ~1.0% CER, and ~92% of sentences reproduced exactly**.

**Homophone vs non-homophone gap (the core research question):** the gap is **small but inverted relative to expectation**. On WER the homophone subset is slightly *better* (1.83% vs 2.95%), and on Exact Match it is also slightly *better* (92.56% vs 90.61%). So at this scale homophone-containing sentences are **not harder** in any reported metric — the language model resolves most ambiguity from context.

### Decoding-strategy comparison (same checkpoint, same 949 rows)

| Decoding | WER | EM | File |
|---|---|---|---|
| Greedy | 3.53% | 86.51% | `p2t_lora_checkpoints_dedup/predictions.csv` |
| **Beam-5 (headline)** | **2.09%** | **91.99%** | `p2t_lora_checkpoints_dedup/predictions_beam5.csv` |

Beam-5 reduces WER by ~1.4 pp absolute and raises exact match by ~5.5 pp absolute over greedy on the dedup set. Both files share the epoch_3 adapter; only decoding differs.

## Key settings

- **LoRA:** rank **r = 48**, alpha = 16, dropout = 0.1, target modules = q/k/v/o
  attention projections (~<1% of parameters trainable).
- **Quantization:** 4-bit NF4, double-quant, fp16 compute (GTX 1080 / Pascal).
- **Prompt format:** `Phonemes: <ARPAbet>\nText: <sentence><eos>`; cross-entropy
  computed only on the sentence (prefix masked with -100).
- **Loss:** cross-entropy + 0.1 × contrastive (cosine hinge, margin 0.5) on
  homophone examples.
- **Optimiser:** AdamW, lr 2e-4, linear warmup, grad-norm clip 1.0, gradient
  accumulation (effective batch ≈ 4).
- **Training schedule:** 3 epochs.
- **Decoding (eval, headline beam-5):** `num_beams=5`, `max_new_tokens=34`,
  `repetition_penalty=1.3`, `no_repeat_ngram_size=3`, first-line extraction applied.
- **Decoding (eval, greedy twin):** `do_sample=False`, `max_new_tokens=24`,
  `repetition_penalty=1.3`, `no_repeat_ngram_size=3`.
- **Metrics:** WER, CER, BLEU-4, Exact Match (via `jiwer` / `sacrebleu`),
  reported overall and stratified by homophone membership.

## Honest caveats to state alongside the numbers

1. **Val is only 949 sentences** (after dedup) — residual CI on error-rate style metrics is non-trivial. Directionally stable; tail counts (e.g. per-phoneme confusions) carry real uncertainty.
2. **No separate test set has been scored yet.** Test is defined (rows 46,922–48,164, 1,243 sentences) but never touched. Headline numbers above are val, not test.
3. **No speaker-disjoint split is possible** with this 2-column CSV (no speaker
   IDs), so train/eval speaker overlap cannot be ruled out — a real risk of
   inflated scores.
4. **No ablation yet.** These numbers don't isolate what the contrastive loss
   contributed; there is no "without-contrastive" baseline run to compare
   against, so we can't yet claim the contrastive mechanism caused the small / inverted homophone gap.
5. **Hard-negative implementation gap:** the contrastive negative currently
   places substituted *word text* in the phoneme slot rather than a re-derived
   phoneme sequence, so the contrastive signal may not be doing exactly what's
   intended (fix identified).
6. The other checkpoint folders (`dryrun_checkpoints/overnight_llama32/test_llama32*`, the Qwen run) are small sanity/plumbing tests (n = 3–40, WER 80–100%) and are **not** results.

## Appendix — files present on disk (do not confuse)

| File | Rows | Decoding | WER | EM | Notes |
|---|---|---|---|---|---|
| `p2t_lora_checkpoints_dedup/predictions_beam5.csv` | 949 | beam-5 | 2.09% | 91.99% | **Headline.** |
| `p2t_lora_checkpoints_dedup/predictions.csv` | 949 | greedy | 3.53% | 86.51% | Same checkpoint, decoding comparison only. |
| `p2t_lora_checkpoints_full/predictions.csv` | 1,082 | beam-5 | 1.81% | 93.53% | **Contaminated** — includes 133 verbatim-in-train duplicates; inflated. Not the headline. |
