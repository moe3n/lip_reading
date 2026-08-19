# Chapter 2. Introduction

## 2.1 Background and context

Lip-reading, also called visual speech recognition, is the task of recovering spoken language from the movement of a speaker's face when the audio signal is absent or unusable. It has practical value in noisy environments where microphones fail, in assistive technology for people with hearing or speech impairments, in silent-speech interfaces, and in forensic and surveillance settings where only video is available (Chung et al., 2017; Afouras et al., 2018). Progress has been driven by large public datasets of natural speech, among which the Oxford-BBC Lip Reading Sentences 2 corpus (LRS2) is among the most widely used: it contains thousands of sentences from British television, spoken by many talkers under unconstrained conditions, which makes it a realistic benchmark for open-vocabulary lip-reading (Chung et al., 2017).

A modern automated lip-reading system is usually built as a pipeline of two stages. The first is a visual front-end that watches the face and predicts a sequence of speech units, most often phonemes, the elementary sounds of a language, or visemes, the visually distinguishable mouth shapes. The second converts that unit sequence into readable text. Recent work from the research group behind this project argues that the second stage, the conversion from phonemes to text, is the main bottleneck on sentence-level accuracy (El-Ghomari et al., 2024): even when the front-end recognises the phonemes well, a weak decoder loses much of that accuracy when it produces the final words.

Two developments have made this second stage a promising place to improve. The first is the rise of large language models (LLMs), neural networks trained on very large text corpora with strong knowledge of how words combine into sentences. Because the phoneme sequence for a sentence encodes it almost completely, apart from ambiguities such as homophones and spelling, converting phonemes to text is close to a translation problem, the kind of sequence-to-sequence task at which language models excel (Raffel et al., 2020). The second is parameter-efficient fine-tuning, in particular Low-Rank Adaptation (LoRA) and its quantised variant QLoRA, which adapt a large pretrained model to a new task by training a small number of extra parameters rather than the whole network (Hu et al., 2021; Dettmers et al., 2023). Together these make it feasible to adapt a multi-billion-parameter language model to phoneme-to-text conversion on a single graphics card.

Prior work in this group established the value of this direction. An initial study compared T5 and GPT-2 for phoneme-to-text translation on LRS2 and found the encoder-decoder T5 clearly superior (El-Ghomari et al., 2024). A follow-up fine-tuned Flan-T5 Large with LoRA to reach 85.11% sentence-level exact match on the LRS2 test set with a 3.23% word error rate (Hossain et al., under review); it also introduced a progressive phoneme preprocessing strategy comparing several input formats based on special tokens and stress markers, and reported an error analysis in which homophone confusions were a very small share of the mistakes. This dissertation continues that line of work with a different family of language model and a wider set of questions.

The problem matters on three levels. Practically, a stronger phoneme-to-text stage raises the accuracy of the assistive and silent-speech applications that motivate lip-reading, at the point current evidence identifies as the bottleneck. Academically, the stage has usually been studied embedded inside a larger multimodal system, so its behaviour as an independent module, how it fails and how it responds to imperfect input, is not well characterised. Technically, decoder-only language models and parameter-efficient fine-tuning make it possible to bring a large pretrained model to this task on modest hardware, opening design choices that earlier encoder-decoder studies did not examine.

## 2.2 Problem statement

The phoneme-to-text decoder in a lip-reading pipeline has to be accurate, evaluated without data leakage, and robust to the imperfect input a real visual front-end produces. Three gaps remain in the existing work on this stage.

The first gap concerns the choice of language model. Existing phoneme-to-text studies on LRS2 used encoder-decoder models such as T5 and Flan-T5. Decoder-only language models, of which the Llama family is a prominent open example (Grattafiori et al., 2024), now match or exceed encoder-decoder models on many generation tasks but have not been evaluated for this task on this corpus. Whether a decoder-only model can match or beat the encoder-decoder result is an open question.

The second gap concerns leakage between training and evaluation. LRS2 is broadcast material, and broadcast speech repeats stock phrases, so a sentence can appear in both the training and evaluation portions; a model that has memorised it scores a perfect result that does not reflect genuine ability. Existing reports of high accuracy on this corpus do not quantify how much of the score comes from such overlap, so the true difficulty of the task is unclear.

The third gap concerns robustness. A decoder trained only on perfect phoneme transcriptions has never seen the errors a visual front-end makes, the wrong, missing, and spurious phonemes. In a complete system the inference-time input contains exactly those errors, so a decoder that performs well only on flawless input may perform poorly in deployment. Whether a decoder degrades gracefully under such corruption, and whether training on deliberately corrupted input improves its robustness, has not been studied for this task.

Underlying all three is a working assumption, common in the lip-reading literature and present in the earlier framing of this problem, that homophones, words with the same pronunciation but different spelling and meaning, are a central source of error. If it holds, effort should go toward handling homophones. Whether homophones are in fact the dominant error type for a fine-tuned decoder has had only limited testing, and if they are not, attention is better directed elsewhere. This project tests the assumption empirically rather than building a method around it.

## 2.3 Aim and objectives

The central hypothesis is that a decoder-only large language model, adapted to phoneme-to-text conversion by parameter-efficient fine-tuning, can match or exceed the accuracy of published encoder-decoder models on LRS2; that its accuracy on a deduplicated evaluation is lower than a headline figure on the standard split, because of sentence overlap between training and evaluation; and that training on deliberately corrupted phonemes improves its robustness to imperfect input at a small cost to clean-input accuracy.

The aim of this project is to develop and evaluate a decoder-only large language model for phoneme-to-text conversion in lip-reading, to establish its accuracy on a deduplicated evaluation of LRS2, to measure and improve its robustness to imperfect input, and to characterise the errors it makes.

The objectives that follow from this aim are:

1. To establish a baseline by evaluating a pretrained Llama-3.2-3B language model on phoneme-to-text conversion with prompting alone, without any training.

2. To fine-tune the same model with QLoRA on the full LRS2 training set and measure its accuracy against the prompting baseline and the published encoder-decoder result.

3. To audit the dataset for overlap between training and evaluation sentences, and to report accuracy on a deduplicated evaluation set that removes that overlap.

4. To measure the fine-tuned decoder's robustness to corrupted phoneme input, and to test whether noise-augmented training improves it.

5. To conduct a structured error pattern analysis of the decoder's mistakes, and to test whether homophones are the dominant error type.

## 2.4 Research questions

The project is organised around four research questions that follow directly from the objectives:

RQ1. Does fine-tuning a decoder-only large language model outperform prompting the same model for phoneme-to-text conversion, and how does it compare with published encoder-decoder results on LRS2?

RQ2. How much of the measured accuracy reflects genuine generalisation once overlap between the training and evaluation sentences is removed?

RQ3. How robust is the fine-tuned decoder to the wrong, missing, and spurious phonemes a visual front-end produces, and does training on deliberately corrupted input improve that robustness?

RQ4. What error patterns dominate the decoder's mistakes, and are homophones the main challenge?

## 2.5 Contributions

This dissertation makes the following contributions to the phoneme-to-text stage of lip-reading.

It shows that a decoder-only language model, Llama-3.2-3B, fine-tuned with QLoRA, reaches high sentence-level accuracy on LRS2, and it places this result against both a prompting baseline of the same model and the published Flan-T5 encoder-decoder result.

It contributes an audit of training-to-evaluation sentence overlap on LRS2 and reports a deduplicated accuracy figure, giving an unbiased measure of task difficulty that a headline number computed on the raw split overstates.

It introduces noise-augmented training for this task, in which a fraction of the training phonemes are deliberately corrupted, and it quantifies the resulting robustness against a clean-trained model across three corruption types and several corruption rates.

It applies a three-stage error pattern analysis framework to the decoder's mistakes and confirms, with a decoder-only model and an analysis covering both a clean-trained and a noise-augmented decoder, the earlier finding that homophones are a small minority of errors (Hossain et al., under review). This corroboration across model families strengthens the case for directing effort toward the error types that actually dominate rather than toward homophone-specific training.

## 2.6 Dissertation organisation

The remainder of the dissertation follows the structure set out in Chapter 1 (Table 1.2): background and related work in Chapters 3 and 4, the models, evaluation methods, and experimental design in Chapters 5 to 7, the data and results in Chapters 8 to 10, and the discussion and conclusion in Chapters 11 and 12.
