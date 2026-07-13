# Verification audit: full-corpus LoRA run (p2t_lora_checkpoints_full)

Date: 12 July 2026
Scope: read-only inspection of the pipeline, the completed full-corpus fine-tuning
run, and its reported results. Nothing was modified; every number below was
recomputed from the saved artifacts and the corpus CSV.
Run under audit: Llama-3.2-3B + LoRA (r=48, alpha 16, targets q/k/v/o), 3 epochs,
sequential split (train = corpus rows 1 to 45,839; validation = rows 45,840 to 46,921).
Reported result: **1.81% WER, 93.53% exact match** on the 1,082-row validation split.

The question asked: are these numbers too good to be true? The audit was run
without a preferred answer; each check below states what was tested, how, and
what was found.

## Summary of verdict

The results are genuine, with one real but small inflation identified and
quantified. The headline 93.53% exact match includes 133 validation sentences
(12.3%) that also appear word-for-word in the training data; the model scores a
perfect 100% on those, which confirms it memorises what it has seen. Removing
them, and additionally removing 212 near-duplicates (within about two word-edits
of a training sentence), the model still scores **91.59% exact match and 2.16%
WER on the 737 validation sentences that have no exact or near duplicate in
training**. That is the honest core number, about 1.9 points below the headline.
Every other check passed: the metrics reproduce independently to four decimal
places, the rows align, no predictions are empty or truncated, the loss curves
show no overfitting turn, and the errors the model does make are phonetically
faithful mistakes on rare words and proper nouns, which is what a real phoneme
decoder gets wrong and not what a broken or leaking evaluation produces.

## Checks performed, in order

### 1. Row-level split integrity

Method: the sequential split code (`build_dryrun_dataframes()` in
`src/p2t_lora/dryrun.py`) slices the corpus at fixed positions, so train and
validation cannot share a row by construction. Verified the boundaries directly
against the corpus and confirmed `predictions.csv`'s 1,082 targets equal the
corpus validation slice row-for-row, in order.

Result: pass. The test split (last 1,243 rows) is referenced nowhere in the
training or evaluation path and remains untouched.

### 2. Metric recomputation with independent code

Method: recomputed WER and exact match from `predictions.csv` using a separate
normalisation and direct jiwer calls, without importing the project's
`metrics.py`.

Result: WER 1.8066%, EM 93.5305%. Identical to the reported values to four
decimal places. No metric bug.

### 3. Output integrity

Method: counted empty or missing predictions; compared mean word length of
predictions against targets (a systematic truncation would show up as shorter
predictions); checked references for line breaks (which the first-line
extraction rule could otherwise clip).

Result: 0 empty predictions out of 1,082. Mean length 5.38 predicted vs 5.37
target words. 0 references contain line breaks. Pass.

### 4. Exact-duplicate leakage between train and validation (the main finding)

Method: the corpus contains repeated sentences (48,164 rows but only 45,455
unique sentences; broadcast fillers like "THANKS FOR WATCHING" appear up to 72
times). A sequential split prevents row overlap but not sentence overlap.
Counted validation sentences appearing verbatim among the 45,839 training
sentences, then split the metrics by that flag.

Result: 133 of 1,082 validation sentences (12.3%) appear verbatim in training,
with identical phoneme sequences. On these the model scores 0.00% WER and
100.00% exact match, a clear memorisation signature. On the remaining 949
never-seen sentences it scores 2.01% WER and 92.62% exact match. The headline
number is therefore inflated by roughly 0.9 points of exact match by this
mechanism.

### 5. Near-duplicate leakage

Method: broadcast speech is formulaic, so a validation sentence can differ from
a training sentence by a single word without being an exact duplicate. Screened
all 949 never-seen validation sentences against all training sentences at
word-edit-distance 1 (insertions and deletions, via deletion-variant hashing)
and at approximately 2 edits including single-word substitutions.

Result: 28 sentences sit within one edit of a training sentence and 212 within
about two edits. The full gradient:

| Validation subset | n | WER | Exact match |
|---|---|---|---|
| Exact duplicate of a training sentence | 133 | 0.00% | 100.00% |
| Near-duplicate (within ~2 word-edits) | 212 | 1.22% | 96.23% |
| Truly novel (no exact or near duplicate) | 737 | 2.16% | 91.59% |
| Headline (all rows pooled) | 1,082 | 1.81% | 93.53% |

Interpretation: performance declines smoothly as sentences get more novel,
rather than collapsing. That is the expected shape for genuine generalisation
with some memorisation on top, not the shape of an evaluation that leaks.

### 6. Overfitting check from the training curves

Method: read `training_history.json` (train and validation loss per epoch,
written during the run).

Result: validation loss fell every epoch (0.0622, 0.0463, 0.0362) while training
loss fell faster (0.185, 0.058, 0.018). No upturn in validation loss, so the
final epoch is the correct one to evaluate. The widening train/val gap is a mild
memorisation signal consistent with finding 4, but validation performance was
still improving when training stopped.

### 7. Qualitative error inspection

Method: read the actual failed predictions rather than only the rates.

Result: the errors are phonetically faithful. Examples: "A LITTLE SAXIFRAGE"
decoded as "A LITTLE SAXOPHONE RAG" (a rare word mis-segmented into common ones
with nearly the same sounds), "WALDORF ASTORIA" as "WALLDORF STORY" (proper
noun), "SQUIRREL" spelled "SQUIRELL". Several counted errors are arguably not
errors at all: "TEN" decoded where the reference has "10" is phonetically
identical and fails only on orthographic convention, meaning the metric is
conservative in places. Exact match also degrades with sentence length (93.4%
at 1 to 4 words down to 87.7% at 10 or more), which is the expected direction.

### 8. Plausibility of the magnitude

Method: internal-consistency arithmetic and comparison with the project's own
history.

Result: 2.16% WER on novel sentences means roughly 97.8% per-word accuracy;
compounding that over the mean 5.4-word sentence predicts roughly 89 to 92%
exact match, matching the observed 91.6%. The numbers are internally
consistent. The jump from earlier runs is explained by known causes: the
5,000-row run (41% EM) used 11 times less training data, rank 8 instead of 48,
and 2 epochs; the older 65%-EM cpt_decoder run additionally suffered evaluation
bugs later fixed (a 24-token generation cap that cut long sentences mid-word,
and no first-line trimming), so it understated its own model. The task itself
is close to deterministic transliteration: stress-stripped ARPAbet phonemes
encode the sentence almost losslessly apart from homophones and spelling, so
very high accuracy after 45,839 training examples is credible for a 3B model.

## What this audit could not verify

- **Decoding configuration of the run.** The generation settings (beam width)
  are not persisted in the run artifacts, so whether beam-5 or greedy decoding
  produced `predictions.csv` cannot be confirmed from disk. This affects how
  the number is annotated in comparisons, not its validity.
- **Pretraining contamination.** LRS2 text is BBC broadcast material and some
  of it may exist in Llama's pretraining data. This cannot be audited from
  here. Mitigating evidence: the identical base model scores 117 to 128% WER
  on this task zero-shot, so pretraining exposure alone provides almost
  nothing; the mapping was learned during fine-tuning.

## Recommendations (no changes made; for discussion)

1. Report the stratified numbers alongside the headline: 91.59% EM / 2.16% WER
   on duplicate-free validation sentences is the defensible thesis claim, with
   the duplicate analysis as supporting methodology. This strengthens rather
   than weakens the result, because it shows the inflation was measured.
2. Final thesis numbers should come from the untouched 1,243-row test split,
   ideally after deduplicating it against the training rows the same way.
3. Future runs should persist the generation settings (beam width, penalties,
   token cap) into the checkpoint directory so evaluations are self-describing.
4. If time allows, run the zero-shot baseline on this same validation split so
   the fine-tuning comparison shares identical evaluation rows end to end.
