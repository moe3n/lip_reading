# Phoneme-to-Text (P2T) Decoder — Full System Walkthrough

A from-first-principles guide to your own codebase, written so you can explain and
defend it. Every claim below is tied to a specific file and function. Where the
code does not show something, it says so explicitly rather than guessing.

Files covered: `model.py`, `dryrun.py`, `data/loader.py`,
`augmentation/hard_negatives.py`, `evaluation/metrics.py`,
`evaluation/error_analysis.py`, `evaluation/contextual_analysis.py`,
`evaluation/llm_judge.py`, `RUNBOOK_real_run.md`.

---

## SECTION 1 — THE BIG PICTURE

### What is P2T decoding, and why does it matter for lip reading?

A lip-reading system can't jump straight from video pixels to a clean English
sentence in one leap. It's usually built in two stages. Stage one (the "front
end", **not in this repo**) watches the mouth and outputs a sequence of
**phonemes** — the atomic sounds of speech, written in ARPAbet (e.g. `TH R UW`
for "through"). Stage two — **your project** — takes that phoneme string and
turns it into the actual written sentence. That second stage is the
**Phoneme-to-Text (P2T) decoder**.

Why split it this way? Because the mouth only carries so much information. Many
distinct sounds look identical on the lips (/p/, /b/, /m/ are the classic
"visemes" — visually indistinguishable). So the phoneme stream you get is noisy
and ambiguous. The P2T decoder's job is to use *language knowledge* — what
English words and sentences are actually plausible — to recover the intended
text from an imperfect phoneme signal. It's the part that contributes
linguistic intelligence, not just visual pattern matching.

### The homophone disambiguation problem, specifically

A **homophone** is two words that sound identical but are spelled differently
and mean different things: MEET / MEAT, SITE / SIGHT, THEIR / THERE. Because
they sound the same, **their phoneme sequences are identical**. So from the
phoneme string alone, the decoder literally cannot tell which one was said —
the information that distinguishes them was never in the audio/visual signal in
the first place. `get_homophones("meet")` in `hard_negatives.py` returns
`["MEAT", "METE"]` precisely because all three share one ARPAbet sequence.

This is the hard core of the problem. A general language model will often guess
the more *frequent* word, or whichever fits a shallow context, and get it wrong.
The thesis claim is that you can train the decoder to be *better than chance* at
this by explicitly teaching it which words are confusable and forcing it to
attend to context to pick the right one.

### How contrastive training with hard negatives helps (the intended mechanism)

A **hard negative** is a wrong answer that's deliberately *almost right* — close
enough that telling it apart from the correct answer is genuinely hard. For a
sentence like "I COULD LABEL THIS AS MEAT", the hard negative is the same
sentence with one homophone swapped: "I COULD LABEL THIS AS MEET"
(`generate_hard_negatives()` in `hard_negatives.py` does exactly this
substitution).

The idea of **contrastive learning** is to give the model two things — the
correct version (the "anchor") and the near-miss (the "negative") — and add a
loss term that *pushes their internal representations apart*. Over training, the
model is pressured to build representations where MEAT-context and MEET-context
don't collapse into the same point, which (in theory) makes it better at
choosing correctly at generation time. The combined objective is
`total_loss = ce_loss + lambda * con_loss` (`cpt_forward()` in `dryrun.py`),
where `lambda` (`contrastive_lambda`, default 0.1) controls how much weight the
contrastive term gets relative to the normal language-modelling loss.

> ⚠️ **Read Section 6, issue #1 before you defend this.** In the code as
> written, the hard negative is built from *substituted English words placed in
> the phoneme slot*, not from a re-derived phoneme sequence. That changes what
> the contrastive loss actually does, and it's the single most important thing
> for you to understand honestly. The mechanism described above is the
> *intent*; the implementation has a gap.

### What is QLoRA, and why was it chosen?

Three ideas stacked together:

- **Full fine-tuning** would mean updating all ~3 billion weights of Llama
  3.2:3B. That needs far more GPU memory than you have (the target hardware is
  GTX 1080s — see the runbook), and risks overfitting on a small dataset.
- **LoRA** (Low-Rank Adaptation) freezes the giant base model and inserts tiny
  trainable "adapter" matrices into a few layers. You train only those — a
  fraction of a percent of the parameters. `load_model_with_lora()` in
  `model.py` does this via `get_peft_model(base_model, lora_config)`.
- **QLoRA** = "Quantized LoRA". On top of LoRA, you also **compress the frozen
  base model to 4-bit** (NF4 format) so it physically fits in less VRAM. That's
  the `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)`
  block in `model.py`.

Net effect: you get to fine-tune a 3B model on a single consumer GPU. The
runbook estimates the quantized model sits at ~1.5–2GB of VRAM. That's the whole
reason QLoRA was chosen — it's the only way this model fits the available
hardware.

### What is the LRS2 corpus, and what does it give you?

LRS2 (Lip Reading Sentences 2) is a well-known BBC-sourced lip-reading dataset —
short spoken sentences from British TV. **Your code never touches the video.**
It consumes a pre-made CSV, `sentphonemepairs_LRS2_original.csv`, which
`load_original_phoneme_text_pairs()` in `loader.py` describes as 48,164
sentences, each paired with a real phoneme transcription, headerless, two
columns. So for your purposes LRS2 provides **(sentence text, phoneme sequence)
pairs** — exactly the input/output the P2T decoder needs. Two companion files,
`sentences_with_homophones_37374.csv` and
`sentences_without_homophones_10790.csv`, pre-label which sentences contain a
homophone-prone word; that labelling is what makes the homophone-vs-non-homophone
comparison possible.

---

## SECTION 2 — THE DATA PIPELINE (one sentence's journey)

### 1. Where the raw data comes from

`build_dryrun_dataframes()` in `dryrun.py` calls
`data_loader.load_original_phoneme_text_pairs()`, which reads
`sentphonemepairs_LRS2_original.csv` (located by `_find_data_dir()` in
`loader.py`, which walks up the folder tree looking for the marker file
`sentences_with_homophones_37374.csv`). The CSV is headerless with two columns
that the loader names `Sentence` and `Phoneme Transcription`.

A docstring note in `dryrun.py` explains *why* this file is used rather than
generating phonemes on the fly: the project's own grapheme-to-phoneme module
(`data/g2p.py`) leaves an unresolved `<UNK>` token on 5.95% of sentences
(British spellings like COLOUR/FLAVOUR that aren't in the standard CMU dict),
whereas this pre-made file has none.

### 2. How phonemes are cleaned — and the `<space>` bug

`clean_phoneme_seq()` in `loader.py` normalises each raw ARPAbet string in four
steps:

1. `re.sub(r"<SOS>|<EOS>", "", seq)` — remove sentence-boundary markers.
2. `re.sub(r"[012]", "", seq)` — strip ARPAbet **stress digits** (e.g. `UW1` →
   `UW`). Stress markers don't help with spelling and would triple the number of
   distinct tokens, so they're dropped.
3. `seq.replace("<space>", " ")` — **this is the bug fix.**
4. Collapse repeated whitespace.

The `<space>` issue (the "" bug you asked about — it's the literal string
`<space>`, a word-boundary marker): the raw phoneme data used a literal token
`<space>` to mark gaps between words. That token does **not exist in the Llama
3.2 tokenizer's vocabulary**, so the tokenizer shattered each one into ~3
sub-tokens. Result: the phoneme prefix nearly doubled in length for zero
phonetic information, and long inputs got truncated. The docstring reports that
`check_token_lengths.py`, measured against the full 48,164-row corpus at
`max_input_len=96`, found this fix dropped the truncation rate from **7.81% to
0.31%** (the runbook later quotes 0.00% after re-running). Replacing `<space>`
with a plain space lets the tokenizer treat word boundaries naturally.

Sentences get their own light cleaning: `clean_sentence()` just uppercases and
strips. So everything downstream is uppercase.

### 3. How a sentence is classified homophone vs non-homophone

Two ways, used at two different points:

- **For dataset partitioning** (`load_stratified_split()` in `loader.py`): it
  loads the two pre-labelled CSVs into sets and checks membership —
  `phoneme_df["sentence"].isin(homo_sentences)`. Anything in neither set is
  dumped into the non-homophone group by default.
- **For deciding whether to build a hard negative** (`CPTDataset.__init__` in
  `dryrun.py`): `is_homo = sentence in homo_set`, where `homo_set` is the set of
  all homophone-sentence strings.

Important subtlety: even if a sentence is *labelled* homophone, the code only
keeps it flagged if a hard negative could actually be generated — `if not negs:
is_homo = False` (more on that in step 5).

### 4. How the train/val split is done — and whether it's "truly stratified"

In `build_dryrun_dataframes()` (`dryrun.py`):

```python
df_homo_sub = homo_df.head(CFG["n_homophone"]).copy()      # first 130 homophone rows
df_non_sub  = non_homo_df.head(CFG["n_non_homophone"]).copy() # first 70 non-homophone rows
df = pd.concat([df_homo_sub, df_non_sub], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle, fixed seed
split = max(1, int(len(df) * 0.8))
return df[:split], df[split:], homo_set
```

Be precise when you describe this, because "stratified" is doing two different
jobs:

- **The selection is stratified.** You deliberately compose the sample as 130
  homophone + 70 non-homophone (≈65/35). That's a controlled mix, not a random
  draw — that part is genuinely stratified by design.
- **The train/val split itself is NOT stratified.** After concatenating and
  shuffling, it's a plain 80/20 slice. Nothing guarantees the 65/35 ratio is
  preserved inside the train half and the val half — that's left to the luck of
  the shuffle. A truly stratified split would use something like
  `train_test_split(..., stratify=homo_label)`. It doesn't. So "stratified
  sampling, random split" is the honest description. (See Section 6, issue #3
  for the deeper leakage problems with `.head()`.)

### 5. What one training example looks like after `CPTDataset.__init__`

Take a real homophone row: sentence = `"I COULD LABEL THIS AS MEAT"`, with its
phoneme string `phonemes`. Inside `CPTDataset.__init__`:

**The prompt format** is a single causal string (a decoder-only model has no
separate encoder/decoder slots, so input and target are one sequence):

```
Phonemes: <phonemes>\nText: I COULD LABEL THIS AS MEAT<eos>
```

The prefix is `prefix = f"Phonemes: {phonemes}\nText:"` and the full string is
`full_text = f"{prefix} {sentence}{tokenizer.eos_token}"`.

**The labels and the -100 masking.** `labels = input_ids.clone()`, then:

```python
labels[:prefix_len] = -100      # don't score the prompt
labels[attention_mask == 0] = -100  # don't score padding
```

`-100` is PyTorch's "ignore this position" value for cross-entropy. Why mask the
prefix? Because you don't want the model rewarded for "predicting" the phonemes
you already handed it — that's input, not something to generate. You only want
the loss computed on the **sentence completion** ("I COULD LABEL THIS AS
MEAT<eos>"). Masking the prefix means the gradient only reflects how well it
generates the *answer*, given the phonemes as context.

There's a careful detail in how `prefix_len` is computed:
`tokenizer(prefix, add_special_tokens=True)`. The comment explains this matters
because some tokenizers (Llama, SmolLM2) prepend a BOS token and some (Qwen2)
don't. If you counted prefix length without special tokens on a BOS-adding
tokenizer, you'd be off by one — masking one token too few, leaking the last
phoneme token in as a training target and shifting the pooling window. So this
line is deliberately matched to how `full_text` is tokenized.

**The negative companion sequence.** For a homophone sentence:

```python
negs = generate_hard_negatives(sentence, max_per_word=1, max_total=1)
neg_text = negs[0]["negative"] if negs else sentence
if not negs:
    is_homo = False
neg_prefix = f"Phonemes: {neg_text}\nText:"
```

So for our example, `generate_hard_negatives` swaps MEAT → MEET (an exact CMU
homophone) and the negative becomes "I COULD LABEL THIS AS MEET". That string is
then placed into the **phoneme position**: `Phonemes: I COULD LABEL THIS AS
MEET\nText:`. It's tokenized to its own `neg_input_ids` / `neg_attention_mask`,
stored alongside the anchor. If no homophone substitution is possible, the
sentence is silently demoted to non-homophone (`is_homo = False`) and gets no
contrastive treatment.

> ⚠️ **This is the crux of issue #1.** Note carefully: the negative's phoneme
> slot contains *English words* ("...AS MEET"), while the anchor's phoneme slot
> contains *ARPAbet phonemes* ("...AE T" etc.). They are not the same kind of
> string. The hard negative is a substituted **sentence**, not a substituted
> **phoneme sequence**. The module docstring at the top of `dryrun.py`
> (point 4) openly flags this as carried over "as-is" from the Flan-T5
> prototype and "worth revisiting." Mechanically, this means the contrastive
> loss is comparing a phoneme representation against an English-text
> representation — see Section 6.

So one finished training item is a dict with: `input_ids`, `attention_mask`,
`labels` (prefix+padding masked to -100), `prefix_len`, `is_homophone`,
`neg_input_ids`, `neg_attention_mask`.

---

## SECTION 3 — THE MODEL (`model.py`)

### 1. The base model

A **decoder-only causal language model** loaded with
`AutoModelForCausalLM.from_pretrained(...)`. The real target is
`MODEL_NAME_TARGET = "meta-llama/Llama-3.2-3B"`. "Causal" means it predicts the
next token left-to-right, each token attending only to tokens before it — there
is no separate encoder, which is exactly why the prompt had to be reformatted
into one sequence (Section 2) and why contrastive pooling had to be redone
(Section 4). The currently-selected `MODEL_NAME_DRYRUN` is
`MODEL_NAME_QWEN = "Qwen/Qwen2.5-0.5B-Instruct"` — a small stand-in (see issue
#5).

### 2. 4-bit quantization (NF4), and why it's needed

Quantization stores each weight in fewer bits. Normally weights are 16- or
32-bit floats; **4-bit** uses 4 bits each — roughly a 4–8× memory cut. **NF4**
("NormalFloat 4") is a 4-bit format designed for the bell-curve distribution
that neural-net weights actually follow, so it loses less accuracy than naive
4-bit. The config also sets `bnb_4bit_use_double_quant=True` (quantizes the
quantization constants too, squeezing out a bit more) and a
`bnb_4bit_compute_dtype` (the precision used for the actual math, even though
storage is 4-bit). It's needed for one reason: **a full-precision 3B model won't
fit in the GPU's VRAM, but a 4-bit one will.**

`USE_4BIT = torch.cuda.is_available()` — quantization only switches on when
there's a CUDA GPU, because the bitsandbytes 4-bit kernels are CUDA-only.

### 3. The LoRA adapter and which layers it targets

`LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]` — the four
projection matrices inside each transformer **attention** block (query, key,
value, and output). LoRA inserts a pair of small low-rank matrices next to each
of these and trains only those. The `LoraConfig` uses `r=8` (the rank — how
"wide" the adapter is), `lora_alpha=16` (a scaling factor), `lora_dropout=0.1`,
`task_type=CAUSAL_LM`, and `bias="none"`. Only the attention projections are
adapted; the big feed-forward (MLP) layers are left frozen. The runbook notes
`lora_r=48` is the intended setting for the full real run.

### 4. "Trainable parameters: X%" — what it means and why it's tiny

`load_model_with_lora()` prints:

```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable (LoRA only): {trainable:>12,}  ({trainable/total*100:.2f}%)")
```

`requires_grad` is the flag that says "this weight will be updated during
training." PEFT freezes the entire base model (`requires_grad=False` on
everything) and flips it back on only for the LoRA adapter weights. So
`trainable` counts just the adapters. The percentage is tiny — typically well
under 1% — because you're training a few small matrices bolted onto a frozen 3B
model. That tiny fraction is the entire point of LoRA: massive memory savings,
fast training, small checkpoints. **This printout is also your verification tool
for issue #6** — if it ever printed 0 trainable params, training would be a
no-op.

### 5. The CPU fallback path — when it activates

Driven by `USE_4BIT = torch.cuda.is_available()` and
`DEVICE = "cuda" if available else "cpu"`. When there's no GPU (your sandbox /
a Mac), `quant_config` stays `None`, the model loads at full precision in
`CPU_DTYPE` (default `bfloat16`), and `base_model.to(DEVICE)` puts it on CPU.
The **same LoRA config** is applied either way, so the full pipeline — data,
loss, training loop, generation, checkpointing — runs end-to-end on CPU. What
the CPU path **cannot** verify is the actual 4-bit bitsandbytes path itself;
that only exists on CUDA. The docstring is explicit: the CPU run "proves the
pipeline mechanics," not the real quantized training.

### 6. The bfloat16 vs float16 issue with GTX 1080s

`bfloat16` and `float16` are both 16-bit float formats but with different
range/precision trade-offs. The catch: **bfloat16 has no hardware support below
NVIDIA compute capability 8.0 (Ampere)**. The uni PC's GTX 1080s are Pascal,
compute capability **6.1** — too old. The original code hardcoded
`bnb_4bit_compute_dtype=torch.bfloat16`, which would have crashed instantly on
the 1080s with `ValueError: Bfloat16 is only supported on GPUs with compute
capability of at least 8.0`. `_select_4bit_compute_dtype()` fixes this: it reads
the GPU's compute capability and returns `bfloat16` only if `major >= 8`,
otherwise `float16`. So on the 1080s it auto-selects `float16`; on a newer card
it would pick `bfloat16` — no code edit needed. Overridable via
`CPT_BNB_COMPUTE_DTYPE`.

---

## SECTION 4 — THE TRAINING LOOP (`dryrun.py`, one step)

### 1. What `cpt_forward()` does

This is the heart of the system. It computes two losses and combines them.

**Cross-entropy loss (`ce_loss`).** One forward pass:

```python
outputs = model(input_ids=..., attention_mask=..., labels=labels,
                output_hidden_states=True)
ce_loss = outputs.loss
```

Because `labels` already has the prefix and padding masked to -100, this loss is
the standard next-token language-modelling loss computed **only on the sentence
completion**. It measures: given the phonemes, how well does the model generate
the correct English words?

**Contrastive loss (`con_loss`).** Computed only if the batch contains at least
one homophone example (`n_homo > 0`):

- **`anchor_vec`** comes *for free* from the same forward pass. The code builds
  a per-example mask that's 1 over each example's phoneme-prefix span and 0
  elsewhere (`positions < prefix_len`), then mean-pools the final-layer hidden
  states over just that span: `anchor_vec = mean_pool(hidden_states,
  prefix_mask)`. This is a single fixed-length vector summarising the phoneme
  prefix's representation.
- **`neg_vec`** needs its **own second forward pass**, because the negative is a
  different input string: `neg_outputs = model(input_ids=neg_input_ids, ...)`
  then `neg_vec = mean_pool(neg_hidden, neg_attn_mask)`.
- They're compared with a margin-based cosine hinge:
  `contrastive_loss()` returns `F.relu(cos_sim - margin).mean()`. In plain
  terms: if the anchor and negative are *more* similar than the margin (0.5)
  allows, you get a penalty proportional to the excess; if they're already far
  enough apart, zero penalty. The gradient pushes their representations apart.

**Combination:** `total_loss = ce_loss + CFG["contrastive_lambda"] * con_loss`
(lambda = 0.1). The CE term teaches it to generate; the contrastive term (at 1/10
weight) nudges representations of confusable inputs apart.

**Why the negative needs its own forward pass but the anchor doesn't.** The
anchor's prefix is *part of* the main sequence you already ran for `ce_loss`.
Because attention is **causal**, the hidden states at prefix positions can only
see earlier prefix tokens — the answer tokens that come afterward can't influence
them. So the prefix's representation is mathematically identical whether you read
it off the full sequence or off a prefix-only pass. That means slicing it out of
the existing pass is free and correct (no look-ahead leakage). The negative, by
contrast, is a *completely different input string* that was never part of that
pass, so the only way to get its hidden states is to actually run it through the
model again. (This mirrors how the old Flan-T5 prototype needed a second
`get_encoder()` call for the negative.)

### 2. Gradient accumulation and why it's used

```python
(total_loss / CFG["grad_accumulation"]).backward()
...
if (step + 1) % CFG["grad_accumulation"] == 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); scheduler.step(); optimizer.zero_grad()
```

With `batch_size=2` and `grad_accumulation=2`, gradients from 2 mini-batches are
summed before a single optimizer update — an **effective batch size of 4**
without ever holding 4 examples in memory at once. You divide the loss by the
accumulation count so the summed gradient has the right scale. This is a standard
trick to simulate larger batches on limited memory. `clip_grad_norm_(..., 1.0)`
caps the gradient size to prevent unstable spikes.

### 3. The learning-rate schedule (linear warmup)

```python
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=CFG["warmup_steps"], num_training_steps=total_steps)
```

The learning rate **ramps up linearly** from 0 to the peak (`2e-4`) over
`warmup_steps` (default 2) optimizer steps, then **decays linearly** back toward
0 over the rest of training. Warmup avoids large, destabilising updates at the
very start when the adapter weights are random; the decay lets it settle. The
optimizer is `AdamW` built *only* on trainable params:
`AdamW([p for p in model.parameters() if p.requires_grad], ...)`.

### 4. What happens during validation

After each epoch:

```python
model.eval()
with torch.no_grad():
    for batch in val_dl:
        total_loss, ce, con = cpt_forward(model, batch)
```

Yes — `model.eval()` switches off dropout and other training-only behaviour, and
`torch.no_grad()` disables gradient tracking (faster, less memory; nothing is
learned). It runs the *same* `cpt_forward` to log val loss / val CE / val
contrastive, purely for monitoring. The model is set back to `model.train()` at
the top of the next epoch.

### 5. What gets saved

```python
model.save_pretrained(CFG["checkpoint_dir"])
tokenizer.save_pretrained(CFG["checkpoint_dir"])
```

Because `model` is a PEFT-wrapped model, `save_pretrained` writes **only the
LoRA adapter weights** (plus their config), not the multi-gigabyte base model.
That's why checkpoints are tiny — the adapter is all that changed. To use the
model later you reload the base model and apply the saved adapter on top.

---

## SECTION 5 — GENERATION & EVALUATION

### 1. How generation works during evaluation

For each validation row (`dryrun.py`, the loop after checkpointing):

```python
prompt = f"Phonemes: {row['phonemes']}\nText:"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
gen = model.generate(**inputs, max_new_tokens=24, do_sample=False,
                     pad_token_id=..., eos_token_id=...,
                     repetition_penalty=1.3, no_repeat_ngram_size=3)
decoded = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()
```

- **The prompt** is just the phoneme prefix — `Phonemes: ...\nText:` — *without*
  the answer. The model continues from there. Slicing
  `gen[0][inputs["input_ids"].shape[1]:]` keeps only the newly generated tokens
  (drops the prompt) before decoding.
- **`max_new_tokens=24`** caps how many tokens it may generate (the answer
  budget). Note this is *smaller* than the training `max_target_len=32` — flagged
  in Section 6.
- **`repetition_penalty=1.3`** down-weights tokens that have already appeared, to
  discourage loops. **`no_repeat_ngram_size=3`** hard-forbids repeating any
  3-gram. The code comment explains these exist to stop the "degenerate 'the dog
  chased the hare...' repetition loops" that a barely-fine-tuned model falls into
  under greedy decoding — runaway repetition can push WER/CER past 100% on
  insertions alone.
- **`do_sample=False`** = **greedy decoding**: always take the single
  highest-probability next token. It's used because it's **deterministic and
  reproducible** — essential for a dissertation where you need the same numbers
  every run. Sampling would introduce randomness into your reported metrics.

### 2. The metrics and what they mean (`metrics.py`)

All computed after `normalise()` (lowercase, strip punctuation, collapse
whitespace):

- **WER (Word Error Rate)** — `(substitutions + deletions + insertions) / number
  of reference words`, via `jiwer.wer`. Lower is better; 0 = perfect. The
  headline metric for this kind of task.
- **CER (Character Error Rate)** — same idea at the character level
  (`jiwer.cer`). More forgiving of small spelling slips; useful when a word is
  *almost* right.
- **BLEU-4** — `sacrebleu` n-gram overlap (up to 4-grams) between hypothesis and
  reference, rescaled to [0,1]. Higher is better. A fluency/overlap measure
  borrowed from machine translation.
- **Exact Match** — fraction of sentences decoded **perfectly** after
  normalisation. The strictest metric.

`stratified_evaluate()` computes all four on three groups: **overall**,
**homophone subset**, and **non-homophone subset**.

**Why the homophone-vs-non-homophone gap is the key result.** `print_results()`
explicitly computes and prints:

```python
wer_gap = (homophone_WER - non_homophone_WER) * 100
em_gap  = (non_homophone_EM - homophone_EM) * 100
"► This gap is the core motivation for CPT training."
```

The thesis isn't really "what's the WER?" — it's "**how much worse is the model
on homophone sentences than on non-homophone sentences, and does contrastive
training shrink that gap?**" A large gap means homophones are genuinely the
bottleneck; the goal of the contrastive mechanism is to narrow it. That gap is
your central experimental quantity.

(There's also a bespoke `homophone_disambiguation_rate()` defined in `metrics.py`,
but **it is not called by `dryrun.py`** — only WER/CER/BLEU-4/Exact Match are
reported in the run. Worth knowing so you don't claim a metric you don't
actually produce.)

### 3. The error pattern analysis (`error_analysis.py`)

After scoring, `error_category_report()` runs. Its job is to answer: *of the
mistakes the model makes, how many are the kind contrastive training is supposed
to fix?*

**How it classifies substitution errors.** `analyze_pair()` runs
`jiwer.process_words()` to align each hypothesis to its reference and extract
every **substitution** (a word swapped for another). Each substitution
`ref_word → hyp_word` is sent to `classify_substitution()`, which buckets it:

- **"Homophone"** — `hyp_word` is an exact CMU-dict homophone of `ref_word`
  (identical phoneme sequence). Cheap dict lookup via `get_homophones()`.
- **"Near-homophone"** — phoneme sequences within edit distance 1 (via
  `get_near_homophones()`, a brute-force scan of the ~125k-word CMU dict, only
  run when the exact check fails).
- **"Other"** — no phonetic relationship, or the word is out-of-vocabulary
  (proper noun, etc.).

**"Phonetically explainable"** means a substitution that's either an exact
Homophone or a Near-homophone — i.e. the model picked a word that genuinely
*sounds like* the right one, as opposed to a random unrelated word.

**What it tells you about whether contrastive training is working.**
`print_error_report()` computes, on the homophone subset, the share of
substitution errors that are phonetically explainable, and prints the decision
rule verbatim:

> High %: contrastive hard-negative tuning is the right lever.
> Low %: errors are likely scale/capacity-driven, not lexical — the real Llama
> run is the more relevant fix, not more hard-negative mining.

In other words: if the model's homophone-subset errors really are confusable
sound-alikes, then sharpening the contrastive mechanism should help. If instead
the errors are mostly unrelated junk, the bottleneck is model size / data scale,
and no amount of hard-negative tuning will fix it. This report is how you tell
those two worlds apart.

**Stage 3 (the deeper "why").** Optionally, each phonetically-explainable
substitution is escalated to find *which* linguistic error category it is:
- **Option 3 — grammar** (`contextual_analysis.py`): uses spaCy dependency
  parsing, but **only** for a fixed list of closed-class words
  (THEIR/YOUR/ITS/MY/OUR/WHOSE). For those, standing in the wrong syntactic role
  is hard proof of a contextual error. The module documents — with tested
  examples — that it deliberately *excludes* THERE/TOO/TWO (spaCy can't reliably
  flag them) and all open-class words like SEE/SEA (the parser happily
  rationalises either spelling). Cheap, deterministic, no model needed.
- **Option 5 — LLM judge** (`llm_judge.py`): asks a language model to classify
  the error (Phonological/Lexical/Contextual/Semantic). **Off by default**
  (`CPT_LLM_ERROR_JUDGE`). The module's own docstring records an honest negative
  result: the CPU dry-run judge (Qwen-0.5B) got *every* test case wrong,
  defaulting to "Semantic" — so its plumbing is validated but its judgements are
  not to be trusted until a real instruct model runs on the GPU.

---

## SECTION 6 — THE "AI MIGHT HAVE MESSED UP" AUDIT

Six things you asked me to scrutinise. For each: severity, what's actually
happening, and a fix.

### Issue 1 — The negative uses substituted TEXT in the phoneme slot
**Severity: HIGH (this is the big one).**

What happens mechanistically: the anchor input is
`Phonemes: <ARPAbet phonemes>\nText:` — the phoneme slot holds sounds like
`AY K UH D L EY B AH L ...`. The negative input is
`Phonemes: I COULD LABEL THIS AS MEET\nText:` — the phoneme slot holds **English
words**. So when `cpt_forward()` pools `anchor_vec` (from a phoneme string) and
`neg_vec` (from an English sentence) and pushes them apart, the model can satisfy
that objective **just by noticing one input is phonemes and the other is
words** — a trivially easy distinction that has nothing to do with homophones.
The "hard negative" isn't hard; it's a different data type.

For this to actually teach homophone disambiguation, the negative's phoneme slot
should contain the **phoneme sequence of the substituted sentence** — but since
MEAT and MEET are homophones, that sequence is *identical* to the anchor's. The
true hard case is "same phonemes, the model must use context to pick spelling,"
and the current code never constructs it. The `dryrun.py` docstring (point 4) and
the `CPTDataset` comment both flag this as inherited "as-is" from the Flan-T5
prototype and "worth revisiting."

**Fix:** build the negative by re-deriving phonemes. Concretely, the negative
input should reuse the **anchor's phoneme prefix** (because the homophone swap
doesn't change the phonemes) and differ only in what the contrastive signal is
attached to — e.g. contrast the representation/likelihood of the *correct
completion* against the *homophone-swapped completion* under the same phoneme
prefix. That makes "MEAT vs MEET given identical phonemes" the actual thing being
learned. This is the most important methodological point to be honest about in
your defense.

### Issue 2 — Cosine similarity on mean-pooled hidden states of a decoder-only model
**Severity: LOW (the math is sound; one small asymmetry).**

Mean-pooling final-layer hidden states into a fixed vector and comparing with
cosine similarity is a standard, defensible sentence-embedding approach. The
specific worry — "could attention in the prefix leak information?" — is **not** a
problem here. Attention is causal: prefix-position hidden states attend only to
*earlier prefix tokens*, never to the answer tokens that follow. So the anchor's
prefix vector carries no look-ahead from the target, and the docstring's claim
that slicing it from the full pass equals a prefix-only pass is correct.

The one real (minor) asymmetry: the **anchor** is pooled over *only the phoneme
prefix span* (`prefix_mask`), while the **negative** is pooled over its *entire*
sequence including the `Phonemes:`/`Text:` scaffolding (`neg_attn_mask`). They're
not pooled over strictly comparable spans. Combined with issue #1, the two
vectors differ for boring structural reasons, not semantic ones.

**Fix:** pool the negative over the same kind of span as the anchor (its phoneme
slot only), and address issue #1 so the comparison is semantically meaningful in
the first place.

### Issue 3 — Simple 80/20 split; speaker/temporal/show leakage
**Severity: HIGH (for any generalisation claim).**

Three distinct problems:

1. **No speaker control is even possible from this data.** LRS2 has speaker and
   video IDs, but `sentphonemepairs_LRS2_original.csv` is **two columns only**
   (sentence, phonemes) — `llm_judge.py`'s docstring explicitly notes there's "no
   video-id, clip-sequence, speaker-id, or timestamp column." So you **cannot**
   guarantee — or even check — that the same speaker doesn't appear in both train
   and val. Speaker overlap is likely and undetectable with this file.
2. **`.head()` causes temporal/show clustering.** `homo_df.head(130)` and
   `non_homo_df.head(70)` take the **first** rows in original BBC-Oxford order.
   Sentences near each other in that order plausibly come from the same
   broadcast/episode. The shuffle happens *after* selection, so train and val are
   drawn from the same narrow slice of the corpus — increasing the chance that a
   val sentence comes from the same show as a train sentence.
3. **The split isn't statistically stratified** (Section 2.4) — the homo/non
   ratio in train vs val is left to chance.

**Fix:** if/when a richer corpus file with speaker/video IDs is available, do a
**speaker-disjoint** split (no speaker in both halves) and a stratified
train/val split (`stratify=` on the homophone label). At minimum, draw the
sample with `.sample(random_state=...)` across the full corpus instead of
`.head()`, and state the speaker-overlap limitation explicitly in the thesis.

### Issue 4 — `repetition_penalty=1.3`, `no_repeat_ngram_size=3` suppressing valid repeats
**Severity: LOW–MEDIUM.**

`no_repeat_ngram_size=3` forbids repeating any **3-word** sequence. Natural short
LRS2 sentences rarely contain a legitimately repeated trigram, so the risk is
small — but not zero (e.g. emphatic or list-like utterances). `repetition_penalty
=1.3` mildly discourages *any* already-used token, which could nudge against
genuine repeats like "very very" or "no no no." The deeper concern is
**methodological**: these are decoding band-aids for an undertrained model, they
differ from training-time conditions, and they alter WER/CER in ways not strictly
attributable to what the model learned — so cross-model comparisons must keep
them fixed.

A second, related mismatch: **`max_new_tokens=24` at generation vs
`max_target_len=32` at training.** A reference longer than 24 tokens gets cut off,
inflating WER through forced deletions independent of model quality.

**Fix:** report metrics both with and without these penalties (or with milder
settings like `repetition_penalty=1.1`) to show they aren't carrying the result;
and raise `max_new_tokens` to at least match `max_target_len` (32) so long
references aren't truncated.

### Issue 5 — Qwen2.5-0.5B stand-in vs the real Llama 3.2:3B
**Severity: MEDIUM (fine for plumbing, invalid for results).**

`MODEL_NAME_DRYRUN` is currently `Qwen/Qwen2.5-0.5B-Instruct`. Differences that
matter:
- **6× fewer parameters** (0.5B vs 3B) and far less capability — any accuracy
  number from the dry run is meaningless as a result.
- **Instruct vs base.** Qwen-0.5B is instruct-tuned; the real target
  `meta-llama/Llama-3.2-3B` is a **base** model (flagged in `llm_judge.py`). They
  behave differently on zero-shot prompting.
- **Tokenizer differences.** Qwen2 doesn't prepend BOS; Llama does. The code
  handles this (the `add_special_tokens=True` `prefix_len` logic, the `[PAD]`
  token), but it's a real behavioural difference and a place bugs could hide when
  you swap models.
- **No 4-bit path is exercised** on CPU at all (issue covered in Section 3.5).

The risk: the dry run validates **shape** (data flow, masking, loss combination,
loop, checkpoint, generation), not **outcomes**. The whole codebase is honest
about this — but don't let a reader mistake dry-run numbers for findings.

**Fix:** none needed for the stand-in's *purpose*; just run Stage 1 of
`RUNBOOK_real_run.md` (real model, real 4-bit, small scale) before trusting any
metric, and present only real-run numbers as results.

### Issue 6 — Could the LoRA adapters silently not be training?
**Severity: LOW as written, but one missing safeguard worth noting.**

The wiring looks correct: `get_peft_model()` freezes the base and enables
`requires_grad` on the adapters; the optimizer is explicitly built from
`[p for p in model.parameters() if p.requires_grad]`; and `cpt_forward` calls
`.backward()` on a loss that flows through those adapters. So adapters *are*
trained on the paths shown.

**How to verify** (do this on the real run):
1. Read the printed line `Trainable (LoRA only): N (X%)` — confirm N > 0.
2. Watch the loss: `train_ce` should fall across epochs. A flat loss = nothing
   learning.
3. After training, confirm the saved checkpoint contains non-zero
   `adapter_model` weights, and that they differ from initialisation (e.g. assert
   not all LoRA `B` matrices are still zero).
4. Sanity-check that `model.print_trainable_parameters()` (PEFT's built-in) and
   the manual count agree.

**The one real gap:** `model.py` does **not** call
`peft.prepare_model_for_kbit_training()` before `get_peft_model()`. For 4-bit
training that helper normally (a) casts layernorms / the LM head to fp32 for
numerical stability and (b) enables input-gradient flow needed when gradient
checkpointing is on. You don't use gradient checkpointing, so training won't
*silently break* — but adding `prepare_model_for_kbit_training()` on the CUDA
path is the standard, safer QLoRA recipe and would remove a class of subtle
stability issues on the real run. **Recommended fix:** call it on the 4-bit path
before wrapping with LoRA.

---

## SECTION 7 — THESIS DEFENSE PREP

### Five likely committee questions, with honest answers

**Q1. "Walk me through how a hard negative is actually constructed — and is it
genuinely 'hard'?"**
This is the question to *get ahead of*, because the honest answer exposes issue
#1. Say it plainly: "The hard negative is the correct sentence with one homophone
word substituted — MEAT→MEET — generated by `generate_hard_negatives()` from CMU
dictionary lookups. In the current implementation, that substituted *sentence
text* is placed in the phoneme slot rather than a re-derived phoneme sequence.
Because the true homophone case has *identical* phonemes, the most faithful
version of this experiment contrasts the correct vs swapped completion under the
same phoneme prefix. I've identified that gap and here's how I'd correct it
[issue #1 fix]." Owning this is far stronger than being caught by it.

**Q2. "How do you know your results generalise — could there be data leakage?"**
Honest answer: "Two limitations. First, the phoneme CSV has no speaker or video
IDs, so I can't guarantee speaker-disjoint train/val splits — speaker overlap is
possible and I can't measure it from this file. Second, the dry-run sample is
taken with `.head()` from the start of the corpus, which can cluster sentences
from the same broadcast. For the real run I'd use a speaker-disjoint, stratified
split. I'm reporting the homophone-vs-non-homophone *gap* rather than absolute
accuracy partly because the gap is more robust to these issues than a raw number."

**Q3. "Why QLoRA and not full fine-tuning, and what do you lose by quantizing?"**
"Hardware — the target GPUs are GTX 1080s; a full-precision 3B model doesn't fit.
QLoRA (4-bit NF4 base + LoRA adapters on the attention projections) gets resident
VRAM to ~1.5–2GB. What I lose is some precision from 4-bit quantization and
expressiveness from only adapting attention layers (not the MLPs), at rank 8 in
the dry run, 48 for the full run. The trade is justified because it's the only
way to fine-tune this model on this hardware, and LoRA is well-established as
near-full-fine-tuning quality for adaptation tasks."

**Q4. "Your contrastive loss operates on mean-pooled hidden states of a
decoder-only model — is that valid, given there's no encoder?"**
"Yes. Llama has no encoder, so I pool the model's own final-layer hidden states
over the phoneme-prefix span. Because attention is causal, those prefix hidden
states never see the answer tokens, so there's no look-ahead leakage — pooling
from the full forward pass is identical to a prefix-only pass, which lets me get
the anchor vector for free and only pay a second forward pass for the negative.
The one refinement I'd make is pooling the negative over a comparable span to the
anchor."

**Q5. "Most of your runs use a 0.5B stand-in model on CPU. What have you actually
demonstrated?"**
"The dry run demonstrates that the full pipeline is correct and runs end-to-end —
prompt formatting, prefix label-masking, LoRA injection, the combined
CE+contrastive loss, training loop, checkpointing, generation, and the stratified
evaluation plus error-pattern analysis. It does **not** demonstrate accuracy
results; a 0.5B instruct model can't stand in for a 3B base model, and the 4-bit
path only exists on CUDA. The real numbers come from the documented Stage 1/Stage
2 GPU runs in the runbook. I deliberately separated 'is the machinery correct'
from 'what does the real model score.'"

### Limitations to acknowledge upfront (before you're asked)

1. **The hard-negative construction (issue #1)** doesn't yet realise the
   identical-phoneme contrastive case it's designed for. Acknowledge it as a
   known methodological gap with a concrete fix.
2. **No speaker-disjoint split is possible** with the current two-column phoneme
   file; speaker overlap can't be ruled out or measured.
3. **Reported dry-run figures are pipeline validation, not results** — the real
   model/4-bit run is required for any accuracy claim.
4. **Evaluation decoding uses repetition penalties and a 24-token cap** that can
   move WER/CER independently of model quality; results should be shown to be
   robust to these.
5. **The LLM error-judge (Stage 3 Option 5) is unvalidated** on small models (its
   own docstring documents it failing every test case), and `prepare_model_for_
   kbit_training()` is not called on the QLoRA path — a standard-recipe omission
   to fix before the real run.
6. **`homophone_disambiguation_rate()` exists but is unused** — don't claim it as
   a reported metric; the run reports WER/CER/BLEU-4/Exact Match only.

Framing these as *your own findings from auditing AI-generated code* is a
strength: it shows you understand the system well enough to know where it's weak,
which is exactly what a committee wants to see.
