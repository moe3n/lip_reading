# P2T Decoder — Results Summary (for supervisor)

**Model:** `meta-llama/Llama-3.2-3B` (decoder-only) fine-tuned with **QLoRA**
(4-bit NF4 base + LoRA adapters).
**Task:** Phoneme → English sentence decoding (P2T) on the **LRS2** corpus, with
contrastive hard-negative training aimed at homophone disambiguation.
**Run:** full-corpus run, logged 27 Jun 2026 (`dryrun_checkpoints/overnight_llama32/`).

## Dataset

| Split | Sentences | Notes |
|---|---|---|
| Full corpus | 48,164 | LRS2 sentence–phoneme pairs (`sentphonemepairs_LRS2_original.csv`) |
| Train | ~38,531 (80%) | not separately logged; inferred from the 80/20 split |
| Eval / validation | **9,633 (20%)** | the set scored below |
| — Homophone subset | 7,511 | sentences containing ≥1 homophone-prone word |
| — Non-homophone subset | 2,122 | |

Split = random 80/20 of the shuffled corpus (seed 42). **There is no separate
held-out test set** — "eval" is the 20% validation split.

## Results (eval set, n = 9,633)

| Subset | n | WER ↓ | CER ↓ | BLEU-4 ↑ | Exact Match ↑ |
|---|---|---|---|---|---|
| **Overall** | 9,633 | **9.55%** | **5.33%** | **0.837** | **65.45%** |
| Homophone | 7,511 | 9.32% | 5.32% | 0.840 | 64.67% |
| Non-homophone | 2,122 | 10.78% | 5.40% | 0.810 | 68.24% |

**Headline:** the fine-tuned model decodes phonemes to text well — **9.5% WER,
5.3% CER, and ~65% of sentences reproduced exactly**.

**Homophone vs non-homophone gap (the core research question):** the gap is
**small and mixed**, not large. On WER the homophone subset is actually slightly
*better* (9.32% vs 10.78%); on Exact Match it is slightly *worse* (64.67% vs
68.24%). So at this scale homophone-containing sentences are **not dramatically
harder** for the model than other sentences.

## Key settings

- **LoRA:** rank r = 8, alpha = 16, dropout = 0.1, target modules = q/k/v/o
  attention projections (~<1% of parameters trainable). *(Note: the runbook
  proposed r = 48 for the full run; the saved adapter used r = 8.)*
- **Quantization:** 4-bit NF4, double-quant, fp16 compute (GTX 1080 / Pascal).
- **Prompt format:** `Phonemes: <ARPAbet>\nText: <sentence><eos>`; cross-entropy
  computed only on the sentence (prefix masked with -100).
- **Loss:** cross-entropy + 0.1 × contrastive (cosine hinge, margin 0.5) on
  homophone examples.
- **Optimiser:** AdamW, lr 2e-4, linear warmup, grad-norm clip 1.0, gradient
  accumulation (effective batch ≈ 4).
- **Decoding (eval):** greedy (`do_sample=False`), max_new_tokens = 24,
  repetition_penalty = 1.3, no_repeat_ngram_size = 3.
- **Metrics:** WER, CER, BLEU-4, Exact Match (via `jiwer` / `sacrebleu`),
  reported overall and stratified by homophone membership.
- *(Training epochs / batch size for this specific run are not stored in the
  checkpoint; script defaults are 2 epochs, batch 2 × accum 2.)*

## Honest caveats to state alongside the numbers

1. **No separate test set** — results are on the 20% validation split, so they
   may modestly overstate true generalisation.
2. **No speaker-disjoint split is possible** with this 2-column CSV (no speaker
   IDs), so train/eval speaker overlap cannot be ruled out — a real risk of
   inflated scores.
3. **No ablation yet.** These numbers don't isolate what the contrastive loss
   contributed; there is no "without-contrastive" baseline run to compare
   against, so we can't yet claim the contrastive mechanism caused the small
   homophone gap.
4. **Hard-negative implementation gap:** the contrastive negative currently
   places substituted *word text* in the phoneme slot rather than a re-derived
   phoneme sequence, so the contrastive signal may not be doing exactly what's
   intended (fix identified).
5. The other checkpoint folders (`test_llama32*`, the Qwen run) are small
   sanity/plumbing tests (n = 3–40, WER 80–100%) and are **not** results.
