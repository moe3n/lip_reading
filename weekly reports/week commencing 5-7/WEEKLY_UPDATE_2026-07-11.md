# Weekly update: LoRA fine-tuning vs. zero-shot prompting

Project: homophone-aware P2T decoding for lip-reading
Date: 11 July 2026
Corpus: LRS2, sentphonemepairs_LRS2_original.csv

## Summary

This week compares two ways of getting from ARPAbet phonemes to English text on the same ~5,000-sentence slice of LRS2: fine-tuning Llama-3.2-3B with LoRA, and prompting the same base model with no training at all. The fine-tuned model reaches 19.96% WER and 41.2% exact match. The zero-shot model, run both with cleaned phonemes and with the raw phoneme string (markers and stress digits intact), stays above 100% WER in both cases and gets almost nothing exactly right. A detailed error breakdown for both is included below. A third leg, a small GRU model trained from scratch with no LLM involved at all, is planned but not run yet, so it is left out of this update and will be added once it exists.

## 1. LoRA fine-tuning results

Base model: Llama-3.2-3B, 4-bit QLoRA. LoRA rank 8, alpha 16, dropout 0.1, adapters on the attention projections only (q/k/v/o). Trained on 4,000 sentence pairs, evaluated on a held-out 1,000, all drawn from a homophone-stratified ~5,000-row pool (about 79% homophone-containing, matching the ratio used for the zero-shot run below so the two are on comparable ground). The contrastive hard-negative loss used in the earlier full-corpus run is switched off here: it turned out to be pushing apart two representations that are frequently phoneme-identical for true homophones, which made the objective meaningless rather than helpful. This run is a plain cross-entropy fine-tune.

| Subset | n | WER | CER | BLEU-4 | Exact match |
|---|---|---|---|---|---|
| Overall | 1,000 | 19.96% | 11.50% | 0.673 | 41.20% |
| Homophone | 791 | 19.26% | 11.30% | 0.682 | 41.85% |
| Non-homophone | 209 | 23.86% | 12.52% | 0.604 | 38.76% |

Homophone-containing sentences score slightly better here, on every metric. That is the opposite of what the "homophones are harder" hypothesis predicts, and it echoes the small, inconsistent gap seen in the earlier full-corpus run (9.32% vs. 10.78% WER, contrastive loss active). Whatever is closing that gap, it does not look like it depends on the contrastive mechanism, since this run does not have one.

## 2. Zero-shot results and analysis

Same base model, same 5,000-row pool, no training. Two prompt formats were compared: `clean` (phonemes with `<SOS>`/`<EOS>`/`<space>`/stress markers stripped) and `raw` (the untouched phoneme string, with an extended instruction explaining the notation).

| Mode | Subset | n | WER | CER | PER | BLEU-4 | Exact match |
|---|---|---|---|---|---|---|---|
| clean | Overall | 5,000 | 128.33% | 94.47% | 103.41% | 0.0123 | 0.20% |
| clean | Homophone | 3,912 | 127.16% | 95.28% | 104.80% | 0.0123 | 0.23% |
| clean | Non-homophone | 1,088 | 134.81% | 90.35% | 96.56% | 0.0125 | 0.09% |
| raw | Overall | 5,000 | 116.71% | 84.72% | 92.90% | 0.0112 | 0.24% |
| raw | Homophone | 3,912 | 115.62% | 84.82% | 93.34% | 0.0110 | 0.15% |
| raw | Non-homophone | 1,088 | 122.72% | 84.18% | 90.75% | 0.0103 | 0.55% |

Both prompt formats fail in the same basic way: WER above 100% means the model inserts and substitutes more words than the reference contains, not just gets individual words wrong. Neither format is close to usable on its own. Between the two, raw scores lower WER and CER across the board, which was not the expected outcome going in, since raw prompts are considerably longer and were the ones that hit a truncation bug earlier in the project (since fixed). With both formats sitting well past 100% WER, this difference is worth flagging but not worth reading too much into yet; a proper significance check, or repeating the comparison at full corpus scale, would settle whether it holds up.

## 3. Detailed error pattern analysis

Every substitution error, from both the zero-shot runs and the LoRA run, was classified as an exact homophone swap, a near-homophone swap (phoneme edit distance 1), or unrelated ("other"), using the same CMU-dictionary-based classifier throughout. A second, independent pass scores the same outputs by articulatory feature: place of articulation, manner of articulation, and voicing.

| Model | Substitutions | Homophone | Near-homophone | Other |
|---|---|---|---|---|
| Zero-shot, clean | 24,069 | 0.1% | 4.3% | 95.6% |
| Zero-shot, raw | 21,651 | 0.2% | 4.7% | 95.1% |
| LoRA fine-tuned | 987 | 2.6% | 13.5% | 83.9% |

The pattern is consistent across all three: most substitution errors have nothing to do with sound-alike confusion. For the zero-shot models this is expected, since a model producing essentially unrelated sentences will not show a phonetic error pattern at all. The more interesting number is the LoRA model's: even with training, 83.9% of its remaining substitutions are still unrelated to phonetics, though the homophone and near-homophone share is six times higher than in the zero-shot runs (16.1% combined vs. under 5%). That points toward the fine-tuned model's remaining errors being more concentrated on genuinely hard phonetic confusions, rather than the wholesale hallucination seen zero-shot, but the dominant failure mode even after fine-tuning is still something other than homophones.

The articulatory-feature breakdown tells a similar story: manner of articulation (how a sound is produced, such as a stop versus a fricative) is the most commonly mismatched feature everywhere, followed by place of articulation, with voicing mismatched least often.

| Model | Place | Manner | Voicing |
|---|---|---|---|
| Zero-shot, clean | 65.3% | 68.9% | 32.7% |
| Zero-shot, raw | 63.1% | 66.2% | 32.0% |
| LoRA fine-tuned | 49.1% | 53.5% | 26.9% |

The LoRA model is lower on all three, consistent with it simply making fewer classifiable phoneme-level errors overall, not a different error profile. A weighted phoneme error rate (WPER), which scores near-miss phonemes as partial credit instead of a full error, shows the same gap: 87.9% (clean) and 79.0% (raw) for zero-shot against 13.8% for the LoRA model, using the same weighting in all three cases.

## Limitations

These numbers come from a 5,000-row slice of the corpus, not the full 48,164 rows, and not a held-out test set separate from whatever might later be used for tuning. The LoRA run used the code's default rank (8), not the rank 48 used in the earlier full-corpus checkpoint, so it is not a clean comparison against that run. The contrastive mechanism is currently off rather than fixed, so this cannot yet be read as an ablation of the contrastive loss specifically, only as fine-tuning versus prompting.

## Still to come

A direct, non-LLM baseline, a small GRU encoder-decoder trained from scratch on the same ~5,000 pairs, is written but not yet run. Once it has real numbers, it becomes the fourth point of comparison: no prior language knowledge at all, prompting a pretrained model with no training, and fine-tuning that same model. That will be added to this report once available.
