# Deduplicated validation run and decoding comparison

Project: homophone-aware P2T decoding for lip-reading
Date: 17 July 2026
Corpus: LRS2, sentphonemepairs_LRS2_original.csv

## Summary

Last week's full-corpus LoRA run scored 93.53% exact match, but an audit found that figure was inflated by validation sentences that also appear word-for-word in training. This week's run removes those duplicates from validation before scoring, using the same model, the same 45,839 training rows, and the same three epochs. The run also switched from beam search to greedy decoding, which turned out to matter more than expected: on the same rows, greedy scores 86.51% exact match against roughly 92.6% for beam-5. That gap, not the deduplication itself, accounts for most of the drop from last week's headline number.

## 1. Run configuration

Base model: Llama-3.2-3B, 4-bit QLoRA. LoRA rank 48, alpha 16, dropout 0.1, adapters on the attention projections only (q/k/v/o). Trained for 3 epochs on the full 45,839-row training split, unchanged from last week's run. Validation is the same 1,082-row sequential split, minus 133 rows whose sentence text also appears somewhere in the 45,839 training rows, leaving 949 clean rows. The test split (1,243 rows) was not touched and stays held out. Generation used greedy decoding (beam width 1), 34 new tokens max, repetition penalty 1.3, no-repeat n-gram size 3, and the same first-line extraction rule as before.

Training loss and validation loss both fell every epoch (train loss 0.183 to 0.057 to 0.018; val loss 0.071 to 0.045 to 0.039), matching the shape of last week's run closely. The checkpoint this week is not meaningfully different in quality from the one that produced last week's headline figure.

## 2. Results and comparison

| Run | Decoding | n | WER | Exact match |
|---|---|---|---|---|
| Full-corpus (last week) | Beam-5 | 949* | 1.95% | 92.63% |
| Dedup (this week) | Greedy | 949 | 3.53% | 86.51% |

\* Last week's figure is recomputed here from the audit's per-subset breakdown, restricted to the 949 rows this week's clean validation set covers, so both rows score the same data.

| Subset (this week) | n | WER | Exact match |
|---|---|---|---|
| Overall | 949 | 3.53% | 86.51% |
| Homophone | 672 | 3.45% | 86.01% |
| Non-homophone | 277 | 3.77% | 87.73% |

Removing training duplicates on its own would be expected to lower exact match slightly, since those duplicate rows scored 100%. The actual drop is larger than duplicate removal explains by itself, because the decoding setting changed in the same run.

## 3. Cause of the gap

This one has a plain explanation. Beam search keeps five candidate sentences at each generation step and returns the best-scoring one at the end. Greedy decoding commits to a single best next word at every step, with no way back once it has taken a wrong turn early in the sentence. The two runs trained on identical data with near-identical loss curves, so the checkpoints themselves are not meaningfully different. The only variable that changed between the two runs is num_beams: 5 last week, 1 (the default) this week. That is enough on its own to explain a gap of this size. Beam search tends to help most on a lightly-trained decoder that has not yet learned to be fully confident in its own next-token choices, which fits a 3-epoch LoRA adapter on a 3B model.

## 4. Next step

Introduce noise into the training data, most likely through augmentation, and see whether that closes any of the remaining gap or changes the error pattern.
