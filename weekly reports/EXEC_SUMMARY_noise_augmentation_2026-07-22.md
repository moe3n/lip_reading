# Noise-augmented fine-tuning: executive summary

Project: homophone-aware phoneme-to-text decoding for lip-reading
Date: 22 July 2026
Corpus: LRS2, sentphonemepairs_LRS2_original.csv
Author: Md Mahmudul Hasan

## Summary

We fine-tuned Llama-3.2-3B on phoneme-to-text conversion twice: once on clean
ground-truth phonemes, once with half the training inputs deliberately
corrupted. Both runs used identical data, hyperparameters, and decoding. The
noise-augmented model scores 88.73% exact match on clean validation against
91.99% for the clean-trained model, so augmentation cost 3.27 points of exact
match on clean input. Whether it bought anything on corrupted input is not yet
measured. That evaluation is built and ready to run but has not produced
results, so this summary reports a cost without a corresponding benefit. The
robustness numbers are the missing half and are the next thing we will produce.

## 1. Why noise augmentation

The model is trained on ground-truth phoneme transcriptions. In a deployed
lip-reading system the phonemes come from a visual front-end that makes
mistakes, so the input at inference time will not look like the input at
training time. Training only on clean phonemes risks a model that performs well
in evaluation and poorly in the pipeline it is meant to sit in. Mixing clean and
corrupted examples in one training run is long-standing practice in speech
recognition, where it is known as multi-condition training.

## 2. Setup

Base model Llama-3.2-3B, 4-bit QLoRA, LoRA rank 48, alpha 16, dropout 0.1,
adapters on the attention projections only. Three epochs on the 45,839-row
training split. Beam search width 5 at generation, 34 new tokens maximum.
Validation is the 1,082-row split minus 133 sentences that also appear
word-for-word in training, leaving 949 rows. The 1,243-row test split has never
been used and stays held out.

Corruption applies to the phoneme input of training examples only. Validation is
always clean, in both runs, so every number below is measured on identical rows
and stays comparable to our earlier results. Half of training rows are corrupted
and half are left clean. Each corrupted row gets one of three operations at a
rate drawn from 5% to 15%:

| Operation | What it simulates |
|---|---|
| Substitute | front-end heard a different phoneme |
| Delete | front-end missed a phoneme |
| Insert | front-end added a phoneme that was not there |

Substituted and inserted phonemes are drawn from the corpus inventory, so the
model never sees a symbol it would not encounter naturally. Both runs share a
fixed random seed, so they start from identical adapter weights and differ only
in training data. Neither run reads the other's checkpoint.

## 3. Results on clean validation

| Model | n | WER | CER | BLEU-4 | Exact match |
|---|---|---|---|---|---|
| Clean-trained | 949 | 2.09% | 0.98% | 0.967 | 91.99% |
| Noise-augmented | 949 | 2.92% | 1.60% | 0.949 | 88.73% |

Split by whether the sentence contains a homophone-prone word:

| Model | Subset | n | WER | Exact match |
|---|---|---|---|---|
| Clean-trained | Homophone | 672 | 1.83% | 92.56% |
| Clean-trained | Non-homophone | 277 | 2.95% | 90.61% |
| Noise-augmented | Homophone | 672 | 2.55% | 89.29% |
| Noise-augmented | Non-homophone | 277 | 4.10% | 87.36% |

Training and validation loss per epoch. Validation is clean in both runs, so the
gap between the two validation columns is a real difference in clean-input
performance, not an artefact of different evaluation data:

| Epoch | Clean train | Clean val | Noise train | Noise val |
|---|---|---|---|---|
| 1 | 0.185 | 0.071 | 0.395 | 0.087 |
| 2 | 0.057 | 0.045 | 0.186 | 0.067 |
| 3 | 0.018 | 0.039 | 0.085 | 0.062 |

## 4. Interpretation

The cost is real and consistent. Every metric moves the same direction, on both
subsets, by a similar margin. This is not noise in the measurement.

Training loss for the augmented run sits about five times higher at epoch 3
(0.085 against 0.018). That is the expected result of corrupting half the
inputs: some training examples no longer contain enough information to recover
the target sentence, so the loss cannot reach the same floor. It reflects a
harder task, not a failure to learn.

Validation loss declined every epoch in both runs with no upturn, so neither
model overfit and epoch 3 is the right checkpoint to evaluate in both cases.
Both were still improving when training stopped.

The homophone subset continues to outperform the non-homophone subset in both
runs, which is the opposite of what the "homophones are the hard case"
hypothesis predicts. We have now seen this pattern in four consecutive runs. We
do not yet have an explanation and are treating it as an open question rather
than a result.

## 5. What is not yet measured

The purpose of noise augmentation is better performance on corrupted input.
Nothing above measures that. Validation is clean in both runs by design, so
these numbers can only show what augmentation cost, never what it gained.

The robustness evaluation feeds corrupted phonemes to a trained model at
inference time, using the same corruption code as training, across three
operations at 5%, 10%, and 20% rates plus a clean control on the same rows. The
corrupted inputs have been generated and are on disk. The generation pass over
them has not been run, so there are no predictions and no metrics.

Until both models are measured on corrupted input, the honest position is that
augmentation has a known cost and an unknown benefit. The two numbers we need
are how far the clean-trained model falls under corruption, and whether the
augmented model falls less. If the clean-trained model degrades gracefully, the
3.27-point cost is not worth paying. If it collapses and the augmented model
holds, the trade is clearly worth it.

## 6. Notes for comparison with other methodologies

For teams working on the same corpus with a different approach, these details
matter for like-for-like comparison:

The split is sequential and matches the official LRS2 partition sizes: 45,839
train, 1,082 validation, 1,243 test. Reported validation numbers use 949 rows
because we remove sentences that appear verbatim in training. That
deduplication matters more than it sounds. LRS2 contains 48,164 rows but only
45,455 unique sentences, and 133 validation sentences appear word-for-word in
training. Our model scores 100% exact match on those, so leaving them in
inflates the headline by roughly two points. Any evaluation on this corpus that
does not deduplicate is measuring partly memorisation.

Decoding also matters more than expected. On identical weights and identical
rows, beam search at width 5 scores 91.99% exact match against 86.51% for greedy
decoding. A 5.5-point difference from decoding alone is large enough that
cross-method comparisons should state which was used.

## 7. Next steps

Run the robustness evaluation on both checkpoints and fill in the missing half
of the comparison. That is the single blocking item.

After that, a detailed error pattern analysis of the surviving errors, following
the shared three-stage framework: substitution, insertion, and deletion
breakdown, then a phoneme confusion matrix, then articulatory feature analysis.
The residual error count is now small enough that this analysis is tractable in
a way it was not at higher error rates.

Final numbers for write-up should come from the untouched 1,243-row test split,
deduplicated the same way.
