# Chapter 5. Models and Techniques

This chapter describes the models compared in this project and the techniques used to train, adapt, and decode them, and it justifies the design choices. The models fall into three groups: two baselines that establish what the task looks like without a fine-tuned language model, the fine-tuned decoder that is the main system, and the techniques of noise augmentation, decoding, and reproducibility that apply to it. Results are reported in Chapter 9; the figures quoted here are given only where they justify a design decision.

## 5.1 Baseline models

Two baselines frame the fine-tuned system, each answering a different question. The no-language-model baseline shows what the task looks like for a model with no prior knowledge of English; the zero-shot baseline shows what a large language model can do on the task with no training at all.

### 5.1.1 The no-language-model baseline

The first baseline is a small recurrent encoder-decoder trained from scratch, with no pretrained language knowledge. It is a single-layer gated recurrent unit (GRU) encoder paired with a single-layer GRU decoder, with 64-dimensional embeddings, a 128-dimensional hidden state, and no attention mechanism, giving about 158,000 trainable parameters. It reads a phoneme sequence and generates the sentence character by character, using teacher forcing during training and greedy decoding at inference.

This baseline is deliberately minimal. Its purpose is to establish a floor: it must learn the entire mapping from phonemes to English spelling from the training data alone, with none of the linguistic knowledge a pretrained model brings. Its expected near-zero sentence accuracy is the point, showing that the task is not solvable by pattern-matching on this amount of data without prior language knowledge. It is trained on the same 45,839-row split and evaluated on the same held-out test set as every other model, so the floor is measured on equal terms.

### 5.1.2 Zero-shot prompting

The second baseline is the pretrained Llama-3.2-3B model prompted to perform the task with no fine-tuning: the phoneme sequence is placed in a prompt and the model completes it. Two prompt formats were used. The clean format strips the corpus markup, the sentence-boundary tokens, word-boundary marker, and stress digits, giving the model a plain phoneme string with a short instruction. The raw format keeps the full markup and uses a longer instruction explaining the notation, so the model is given every symbol the corpus provides. Comparing the two isolates whether the extra markup helps or hinders a model never trained on it.

The zero-shot baseline measures the untrained ability of the same model that is later fine-tuned, which makes the comparison between them a clean test of what fine-tuning adds. Prompted alone on the full training split, the model reaches near-zero exact match with a word error rate above 100%, so the base model cannot perform the task without training.

## 5.2 The fine-tuned model: Llama-3.2-3B with QLoRA

The main system is the Llama-3.2-3B model adapted to phoneme-to-text conversion by QLoRA fine-tuning.

### 5.2.1 Choice of model

Llama-3.2-3B is a decoder-only language model, using a single stack that predicts each token from the tokens before it, the design of most current general-purpose language models. The prior work on this task used encoder-decoder models, so a decoder-only model tests an open question directly. The task is framed as prompt completion: the phonemes and a short label form the prompt, and the model generates the sentence as the continuation. The 3-billion-parameter size is the largest in the family that fits the available hardware once quantised, balancing capacity against the single-GPU memory budget.

### 5.2.2 QLoRA adaptation

Full fine-tuning, which updates all of the model's weights, was not feasible on the available graphics card and was rejected on that basis. Standard LoRA freezes the pretrained weights and trains small adapter matrices, reducing trainable parameters but still holding the full-precision model in memory, which was also too large. QLoRA addresses both constraints: it quantises the frozen model to four-bit precision, cutting its memory footprint by roughly a factor of four, and trains only the low-rank adapters on top. The four-bit format is NF4 with double quantisation, and the adapters are applied to the query, key, value, and output projections of the attention layers, with rank 48, a scaling factor of 16, and dropout of 0.1. This makes about 1.5% of the parameters trainable and keeps the whole process within a single card's memory.

The obvious risk of four-bit quantisation is a loss of precision that could harm accuracy. Whether that matters for a task as near-deterministic as phoneme-to-text conversion is an empirical question, and Chapter 9 answers it: the quantised model reaches high accuracy, so the precision loss does not prevent it from learning the mapping well.

### 5.2.3 Prompt format and training

Each training example is a single sequence, "Phonemes: P\nText: S", followed by an end-of-sequence token, where P is the phoneme string and S the target sentence. Loss is computed only on the sentence part, with the phoneme prompt and label masked out, so the model learns to generate the sentence given the phonemes rather than to predict the phonemes. Because the base model has no dedicated padding token, one is added and the token embeddings resized to match, which the adapter checkpoints record.

Training runs for three epochs with the AdamW optimiser, a learning rate of 2e-4 with a short warmup, and gradient accumulation to reach an effective batch size within the memory budget. An adapter checkpoint is saved after each epoch, as both insurance against a failed run and a guard against overfitting, since an earlier checkpoint remains available if a later epoch degrades. Here the validation loss fell every epoch, so the final epoch was evaluated.

## 5.3 Noise augmentation

The fine-tuned model as described is trained on clean phoneme transcriptions. Noise augmentation makes it robust to the imperfect phonemes a visual front-end produces, by corrupting a fraction of the training examples so the model learns to recover the intended sentence from imperfect input.

The corruption uses three operations, each standing in for a way a visual front-end fails: substitution replaces a phoneme with a different one (a misheard sound), deletion removes a phoneme (a missed sound), and insertion adds a spurious phoneme (a sound that was not spoken). Substituted and inserted phonemes are drawn from the phonemes that actually occur in the corpus, so the model never sees a symbol it would not encounter naturally.

Two choices govern how much corruption is applied. Half of the training examples are corrupted and half left clean; corrupting every example was rejected because it would remove the model's exposure to clean input and risk the clean-input accuracy multi-condition training is designed to protect. The corruption rate for a corrupted example is drawn from 5% to 15% rather than fixed, so the model cannot calibrate to a single noise level. Corruption is applied to the training input only, with validation and test inputs always clean, so the reported accuracy stays comparable to the clean-trained model. One known simplification is that corruption is applied once when the dataset is built, so a given example keeps the same noise across all epochs; resampling fresh noise each epoch would be stronger augmentation and is noted as future work.

## 5.4 Decoding strategies

At generation time the model produces the sentence one token at a time, and the decoding strategy governs how each token is chosen. Two are compared. Greedy decoding takes the single most probable token at every step. Beam search of width five keeps the five most promising partial sentences at each step and returns the highest-scoring completed one, letting it recover from an early token choice that greedy decoding could not revise.

Decoding is treated as an explicit variable because it was found to matter: on the deduplicated validation set, with the same model, beam search raised exact match from 86.51% to 91.99% and roughly halved the word error rate. A difference this large from decoding alone means any comparison between models must hold the strategy fixed, which is why the beam-versus-greedy setting is recorded for every run.

Three further generation settings are used. A repetition penalty and a no-repeat rule suppress the degenerate loops a lightly trained decoder can fall into, repeating a phrase until the token limit. A maximum of 34 new tokens covers every sentence in the corpus with margin, so no target is cut off. Only the first line of the output is kept, removing any trailing text the model appends after its answer without affecting a correct single-line transcription.

## 5.5 Reproducibility techniques

Reproducibility was built into the runs. A fixed random seed for the data shuffling and adapter initialisation lets a run be repeated exactly and a difference between two runs be attributed to a real change rather than chance; two independent clean-trained runs landed within a fraction of a percent on training loss, confirming the results are stable and not an artefact of a lucky seed. Every generation setting, including beam width and token limit, is recorded with the run, which is what allowed the decoding difference above to be diagnosed. All code, configuration, and results are under version control, and the per-epoch checkpoints let a run be resumed or an earlier epoch re-evaluated without retraining.
