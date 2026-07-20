# Response to supervisor questions (12 July 2026)

Every number and claim below was cross-checked against the code and the corpus
before writing. Where the supervisor's understanding differs from what we
actually did, the difference is stated plainly rather than smoothed over.

---

## Q1. Which split did we evaluate on: validation or test?

**We evaluated on the validation split, not the test split.** The supervisor's
email assumes the numbers are on the LRS2 test split; they are not. This is the
main correction to make.

### What our split actually is

Our corpus file (`sentphonemepairs_LRS2_original.csv`) holds 48,164
sentence–phoneme pairs in original LRS2 order. We slice it sequentially:

| Split | Rows (0-based) | Count | Used for |
|---|---|---|---|
| Train | 0 – 45,838 | 45,839 | fine-tuning |
| **Validation** | **45,839 – 46,920** | **1,082** | **the reported 1.81% WER / 93.53% EM** |
| Test | 46,921 – 48,163 | 1,243 | held out, never touched |

These three counts (45,839 / 1,082 / 1,243) are exactly the official LRS2 main
partition sizes, so our sequential slices reproduce the official LRS2 train,
validation, and test sets. The 1,082 predictions in the CSV are the **full
official LRS2 validation split**, evaluated without any deduplication.

### Why validation and not test

Standard practice: the test split is held out untouched until final reporting,
so that nothing about it can leak into development or model-selection decisions.
We verified in the code that the test rows are referenced nowhere in the
training or evaluation path. Reporting on validation during development, and
reserving test for the final thesis number, is the deliberate choice here.

### Reconciling the counts (1,082 vs the expected 1,079)

The supervisor's arithmetic (1,242 test samples, minus 34 duplicates, minus 129
train-copies, equals 1,079) is about the **test** split. It does not apply to our
number because we reported on **validation** (1,082), which we did not
deduplicate before computing the headline.

Importantly, we reached the same concern independently. Our own leakage audit
(`VERIFICATION_AUDIT_2026-07-12.md`, and the re-runnable
`comparison/verify_run.py`) found that of the 1,082 validation sentences:

- 133 (12.3%) appear verbatim in the training set; the model scores 100% exact
  match on those, a clear memorisation effect.
- A further 212 are near-duplicates (within about two word-edits of a training
  sentence).
- 737 are truly novel (no exact or near duplicate in training).

On the 737 truly-novel validation sentences the model still scores **91.59%
exact match and 2.16% WER**. That is the honest, leakage-free number, about 1.9
exact-match points below the headline. So the supervisor's instinct about
train/eval copies is correct, and we have already measured its effect. Per-row
labels with corpus row numbers are in
`comparison/results/verification/leakage_per_row.csv`.

### Recommendation for the thesis

Final numbers should come from the untouched test split, deduplicated against
training the same way, and reported alongside the stratified validation table so
the memorisation effect is visible rather than hidden.

---

## Q2. Word boundaries: did we keep the `<space>` token, or substitute a marker like `|`?

**We removed the `<space>` token entirely, and did not replace it with any
marker.** No `|`, no `_`, nothing. So yes, the model infers word boundaries on
its own. Confirmed directly against the data.

### What the model actually sees

Raw phoneme string in the file:

```
<SOS> W EH1 N <space> Y UW1 <space> ' R EY1 <space> K UH1 K IH0 NG <space> ... <EOS>
```

After cleaning (`clean_phoneme_seq` in `src/p2t_lora/data/loader.py`):

```
W EH N Y UW ' R EY K UH K IH NG ...
```

The cleaning removes `<SOS>`/`<EOS>`, strips the stress digits (0/1/2), and
replaces each `<space>` with an ordinary space. Since a plain space is already
the separator between phonemes within a word, the word-boundary information is
genuinely gone: the space between the last phoneme of one word and the first of
the next is indistinguishable from the space between two phonemes inside a word.
We verified the cleaned strings contain no `<space>`, no `|`, no `_`, and no
special boundary character of any kind.

### Why we removed it

`<space>` is not a token in Llama 3.2's vocabulary, so the tokenizer shattered
each one into roughly three sub-tokens. That nearly doubled the phoneme-prefix
length while carrying no phonetic information. Measured on the full 48,164-row
corpus, removing `<space>` dropped the input truncation rate at a 96-token
budget from 7.81% to 0.31%. The trade was made for tokenizer efficiency, and it
is documented in the loader's docstring.

### The honest implication (which supports the supervisor's intuition)

The supervisor is right that word boundaries matter. What our result shows is
that the fine-tuned model learns to recover them from the phoneme stream plus
its knowledge of English, rather than needing them marked in the input: it
reproduces correctly-spaced sentences at 93% exact match despite the input
having no boundary markers at all. Whether reintroducing boundaries as a single
dedicated token (for example a `|` added to the tokenizer, so it costs one token
instead of three) would improve accuracy further is a clean ablation we have not
run. It is a good candidate for a follow-up experiment.

---

## Q3. Training configuration

| Setting | Value |
|---|---|
| Base model | meta-llama/Llama-3.2-3B (decoder-only) |
| Adapter | LoRA, rank 48, alpha 16, dropout 0.1, on attention projections q/k/v/o |
| Trainable parameters | ~27.5M (about 1.5% of the model) |
| Quantization | 4-bit NF4, double-quant, fp16 compute (Pascal has no bf16) |
| Learning rate | 2e-4, AdamW, weight decay 0.01, linear warmup then decay, grad-norm clip 1.0 |
| Batch size | 4 (gradient accumulation 1, so effective batch 4) |
| Max sequence length | 128 tokens (96 phoneme-prefix budget + 32 target) |
| Epochs | 3 |
| Loss | plain cross-entropy on the target sentence only (phoneme prefix label-masked; the contrastive term is disabled in this run) |
| Decoding (eval) | beam search width 5, max 34 new tokens, repetition penalty 1.3, no-repeat-ngram 3 |
| GPU | NVIDIA GeForce GTX 1080 (Pascal, 8 GB); at 4-bit the 3B model fits on a single card (the machine has two 1080s available) |
| Training time | ~15.4 hours for 3 epochs (about 5 hours 10 minutes per epoch), from the per-epoch checkpoint timestamps |
| End-to-end | ~16 hours including beam-5 generation on 1,082 validation rows plus error analysis |

### Per-epoch loss (no overfitting turn)

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 0.185 | 0.062 |
| 2 | 0.058 | 0.046 |
| 3 | 0.018 | 0.036 |

Validation loss fell every epoch, so epoch 3 is the right one to evaluate. The
widening gap between train and validation loss is a mild memorisation signal,
consistent with the duplicate finding in Q1, but validation was still improving
when training stopped.

---

## Draft email reply

> Thank you, and thank you for looking at the predictions so carefully. Your
> questions caught two things worth clarifying.
>
> On the split: the numbers are on the LRS2 **validation** split (1,082
> sentences), not the test split. Our corpus file holds the official LRS2 main
> partition in order, so a sequential slice reproduces the official train
> (45,839), validation (1,082), and test (1,243) sets exactly. We report on
> validation during development and are keeping the test split untouched for the
> final numbers, which is why the count is 1,082 rather than a deduplicated test
> count. Your point about train/eval copies is well taken: we ran a leakage
> audit and found 133 of the 1,082 validation sentences appear verbatim in
> training, with the model scoring 100% on those. On the 737 validation
> sentences that have no exact or near duplicate in training, it still scores
> 91.6% exact match and 2.16% WER. We will report that stratified number
> alongside the headline, and take the final figure from the deduplicated test
> split.
>
> On the `<space>` token: we remove it entirely and do not replace it with any
> marker, so the model does infer word boundaries from the phoneme stream. We
> dropped it because it is not in Llama's vocabulary and was expanding into about
> three sub-tokens per boundary, nearly doubling the input length; removing it
> cut our truncation rate from 7.8% to 0.3%. You are right that boundaries
> matter, and it is notable that the model recovers correct spacing at 93%
> exact match without them. Testing a single-token boundary marker is a clean
> ablation we would like to run next.
>
> Training configuration: Llama-3.2-3B with 4-bit QLoRA (rank 48, alpha 16,
> targets q/k/v/o, about 1.5% of parameters trained), learning rate 2e-4 with
> AdamW, effective batch size 4, maximum sequence length 128 tokens, 3 epochs,
> beam-5 decoding at evaluation. It trained in about 15.4 hours on a single GTX
> 1080 at 4-bit. Happy to share the full config or the audit report if useful.
