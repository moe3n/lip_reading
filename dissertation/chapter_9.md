# Chapter 9. Results

This chapter reports the experimental results: the two baselines, the fine-tuned model and its accuracy, the contamination audit and the deduplicated figure, the noise-augmentation and robustness experiments, the effect of the decoding strategy, and a comparison across all runs. The headline figures for the clean-trained and noise-augmented models and the no-language-model baseline are on the held-out test set; the robustness curves and corruption-rate comparisons use a development probe sample and the deduplicated validation set, since those experiments were run during development.

## 9.1 Baselines

Neither baseline can perform the task, which establishes the floor the fine-tuned model is measured against.

The no-language-model baseline, a small recurrent network trained from scratch on the full training set, reaches 0.08% exact match on the held-out test set, a single correct sentence out of 1,243, with a word error rate of 88.1%. With no pretrained language knowledge and only about 158,000 parameters, it cannot learn the mapping from phonemes to English spelling from the training data alone.

The zero-shot baseline, the pretrained Llama-3.2-3B model prompted without any training, reaches about 0.2% exact match with a word error rate above 100%, generating fluent but unrelated text rather than a faithful transcription. The raw and clean prompt formats made little difference, both failing in the same way. The base model, prompted alone, cannot perform the task.

## 9.2 Fine-tuning result

Fine-tuning the same Llama-3.2-3B model with QLoRA transforms the result. On the held-out test set the clean-trained model reaches 88.82% exact match on the standard split and 87.44% after deduplication, with word error rates of 2.47% and 2.73%. On the deduplicated validation set used during development it reaches 91.99% exact match with a 2.09% word error rate. The gap between validation and test is expected, since the validation set guided development while the test set was untouched.

Table 9.1 gives the full test-set metrics for the clean-trained model, split by whether the sentence contains a homophone-prone word.

| Test set | n | WER | CER | BLEU-4 | Exact match |
| --- | --- | --- | --- | --- | --- |
| Standard, overall | 1,243 | 2.47% | 1.12% | 0.955 | 88.82% |
| Standard, homophone | 952 | 2.18% | 0.96% | 0.958 | 88.66% |
| Standard, non-homophone | 291 | 3.91% | 1.84% | 0.934 | 89.35% |
| Deduplicated, overall | 1,091 | 2.73% | 1.24% | 0.951 | 87.44% |

Table 9.1. Held-out test-set results for the clean-trained model, beam-search decoding.

The homophone and non-homophone subsets score within about one point of each other, so sentences containing a homophone-prone word are not markedly harder for the model, which anticipates the error analysis finding that homophones are a minor error source.

## 9.3 Contamination audit and deduplicated evaluation

The exploratory analysis found that roughly one evaluation sentence in eight appears verbatim in training, and the audit confirmed the effect on reported accuracy. On the test set, 152 of the 1,243 sentences appear in training, and removing them lowers exact match from 88.82% to 87.44%, a gap of 1.4 points. On the validation set, 133 of the 1,082 appear in training; the model scores a perfect result on those, and removing them lowers exact match to 91.99%.

The deduplicated figure is the honest measure, since the removed sentences are ones the model may have memorised rather than generalised to. Both are reported: the standard figure remains available for comparison with published work on the raw split, while the deduplicated figure is the headline claim of the model's ability.

## 9.4 Noise augmentation and robustness

The clean-trained model is accurate on clean input but fragile when its input is corrupted, as a visual front-end would corrupt it. Figure 9.1 shows how the two models degrade as an increasing fraction of the input phonemes is corrupted, for each of the three corruption types.

![Figure 9.1. Exact-match accuracy as input corruption increases, for each corruption type. The clean-trained model (red) falls sharply once any corruption is present; the noise-augmented model (blue) declines gradually. Both start from the same point on clean input.](figures/robustness.png)

On clean input the clean-trained model reaches 93% exact match on the probe sample, a small development subsample used only for the corruption sweep, so its clean-input point sits a little above the full-set figure of 88.82%. A 5% corruption rate, one phoneme in twenty, drops it to between 19% and 46% depending on the corruption type, and a 20% rate drops it close to zero. The clean-trained model depends heavily on receiving an exact phoneme sequence.

The noise-augmented model, trained with half of its input phonemes corrupted, degrades far more gently. Under the same 5% corruption it holds between 58% and 81% exact match, and under 20% corruption it still reaches between 20% and 51%. Table 9.2 gives the exact-match figures at the 5% rate, where the difference is already large.

| Corruption at 5% | Clean-trained | Noise-trained |
| --- | --- | --- |
| Substitution | 19.3% | 58.3% |
| Deletion | 33.0% | 66.7% |
| Insertion | 46.0% | 80.7% |

Table 9.2. Exact-match accuracy under 5% input corruption.

The robustness is bought at a small cost to clean-input accuracy. On the held-out test set the noise-augmented model reaches 86.08% exact match on the standard split and 84.14% after deduplication, against the clean-trained model's 88.82% and 87.44%, a difference of about three points, with a word error rate of 3.14% and 3.53% respectively. The deduplicated validation set shows the same gap, 88.73% against 91.99%. For a deployment where the input will be imperfect, this trade favours the noise-augmented model.

## 9.5 Effect of the decoding strategy

The decoding strategy has a large effect on accuracy, which is why it is treated as a controlled variable. On the deduplicated validation set, with the same clean-trained model, greedy decoding reaches 86.51% exact match while beam search of width five reaches 91.99%, a difference of about five and a half points, and beam search roughly halves the word error rate. Beam search recovers from an early token choice greedy decoding could not revise, which matters most on a decoder not yet fully confident in its first choice. Because a difference this large can arise from decoding alone, every comparison between models holds the decoding strategy fixed.

## 9.6 Cross-run comparison

Table 9.3 brings the runs together. The two baselines sit at the floor, unable to perform the task. Fine-tuning raises exact match from near zero to the high eighties on the held-out test set, which is the central result. The clean-trained model exceeds the published encoder-decoder result on the same standard test set, and its honest deduplicated figure still exceeds it. The noise-augmented model trades a few points of clean-input accuracy for a large gain in robustness.

| Model | Exact match | Word error rate | Evaluation |
| --- | --- | --- | --- |
| GRU (no language model) | 0.08% | 88.1% | held-out test |
| Zero-shot Llama-3.2-3B | ~0.2% | above 100% | full corpus |
| Flan-T5 encoder-decoder (published) | 85.11% | 3.23% | standard test |
| LoRA clean-trained (standard) | 88.82% | 2.47% | held-out test |
| LoRA clean-trained (deduplicated) | 87.44% | 2.73% | held-out test |
| LoRA noise-augmented (standard) | 86.08% | 3.14% | held-out test |
| LoRA noise-augmented (deduplicated) | 84.14% | 3.53% | held-out test |

Table 9.3. Cross-run comparison. All fine-tuned and baseline figures are on the held-out test set, except the zero-shot baseline, which is on the full corpus.

The results answer the first three research questions directly. Fine-tuning beats prompting and the no-language-model baseline, and the decoder-only model beats the published encoder-decoder result; the deduplicated figure is lower than the standard by a measured amount, establishing the honest accuracy; and noise-augmented training gives a large improvement in robustness for a small clean-input cost. The fourth research question, the dominant error types, is answered by the error analysis in Chapter 10.
