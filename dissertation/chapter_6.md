# Chapter 6. Evaluation Techniques

This chapter explains how the models are scored. It covers the conventional metrics that measure how much a prediction differs from the reference, the semantic measures that ask whether the meaning is preserved, and the three-stage error analysis framework that moves from counting errors to explaining them. The limitations of each metric are stated, because part of the argument of this project is that no single number describes a phoneme-to-text decoder well.

## 6.1 Conventional metrics

The primary metrics are error rates and an exact-match accuracy. All compare the predicted sentence with the reference after both are normalised to a consistent case and stripped of punctuation, so scoring compares words and sounds rather than formatting.

Word error rate (WER) is the standard measure in speech recognition. It aligns the prediction with the reference by edit distance, counts the substitutions, insertions, and deletions needed to turn one into the other, and divides by the number of reference words; zero means a perfect transcription. It can exceed 100% when the prediction inserts and substitutes more words than the reference contains, which is a useful signal in itself: the model is generating unrelated text rather than a faithful transcription, as the untrained baselines do.

Character error rate (CER) applies the same edit-distance measure at the character level. It is more forgiving than word error rate, since misspelling one letter of a word counts as a small character error but a whole word error. Reporting both separates a model that chooses wrong words from one that mostly makes small spelling slips.

Phoneme error rate (PER) applies the same measure at the phoneme level. Because the model outputs text, it is computed by converting both predicted and reference sentences back to phonemes with a grapheme-to-phoneme tool and aligning those. It measures how far apart the two sentences are in sound, which can differ from their distance in words: two sentences can share few words yet sound similar, or share many words yet differ in a few important sounds.

Exact-match accuracy (EM) is the strictest metric, the fraction of predictions that match the reference exactly after normalisation, so a single wrong word fails the whole sentence. It is the headline metric here because the practical goal is a correct sentence, and it is what the prior work on this task reports, making it the natural point of comparison.

BLEU is a fluency-oriented metric from machine translation. It measures the overlap of short word sequences, up to four words long, between prediction and reference, with a penalty for predictions that are too short. It is reported alongside the error rates as a secondary measure, because a high BLEU indicates fluent, well-formed English even where individual words are wrong. Error rates were computed with jiwer and BLEU with sacrebleu, both standard implementations, so the numbers are comparable with other work using the same tools.

## 6.2 The limits of surface metrics

These metrics measure surface agreement between prediction and reference. They are necessary but not sufficient, and understanding why motivates the rest of the evaluation. Each treats every error as equally serious: a word error rate does not distinguish a harmless substitution, such as a digit for a spelled-out number, from one that changes the sentence's meaning. It also says nothing about why an error occurred or whether the errors follow a pattern, so two models with the same word error rate can fail in entirely different ways a surface metric cannot distinguish. The semantic measures and the error analysis framework in the rest of this chapter address these gaps.

## 6.3 Semantic similarity

Semantic similarity asks whether the prediction means the same thing as the reference, even when the exact words differ. Two measures are used, and a central finding of this project is that they can disagree, so both are reported.

The first is BERTScore, which compares prediction and reference using contextual token embeddings from a pretrained language model rather than exact word matches: each token in the prediction is matched to the most similar token in the reference, and the scores combined into a single figure (Zhang et al., 2020). Because it works at the token level, BERTScore rewards word overlap, so it can report a high score even when the overall meaning has changed, as long as most words are shared.

The second is Sentence-BERT, which encodes each whole sentence into a single vector and measures the cosine similarity between the two (Reimers and Gurevych, 2019). Because it represents the whole sentence rather than matching tokens, it is more sensitive to a change in meaning: a substitution that alters what the sentence says pulls the two vectors apart even if most words are shared. This project uses the sentence-level measure as its primary semantic figure, with the token-level measure for contrast, because it better reflects whether the meaning survived.

The distinction matters because the framework itself warns that a similarity score alone is insufficient: a fluent sentence whose meaning has reversed can still score highly. Using both a token-level and a sentence-level measure, and reporting where they diverge, is this project's response to that warning.

## 6.4 The three-stage error analysis framework

The error analysis follows a three-stage framework adopted within the research group, grounded in the tradition of classifying errors by linguistic type rather than merely counting them (Corder, 1967) and in the structured error taxonomies used to assess machine translation quality (Lommel et al., 2014). It moves from measuring how much error there is, through what kind of error it is, to why it occurred.

Stage 1 is conventional evaluation, reporting the phoneme, word, and character error rates described above. It answers how much the model gets wrong, the stage that surface metrics alone provide, and the framework's premise is that it is necessary but leaves the important questions unanswered.

Stage 2 is phoneme error pattern analysis. It breaks errors down by type, counting substitutions, insertions, and deletions at the word and character level, and examines which sounds the model confuses. This includes a confusion matrix recording, for each reference phoneme, which phoneme the model produced instead, and an analysis by articulatory feature, grouping confusions by place, manner, and voicing. A weighted phoneme error rate, which discounts a substitution between two similar sounds relative to two very different ones, tests whether the errors are near misses or scattered. This stage answers what the model confuses.

Stage 3 is hierarchical error analysis, the most involved. It classifies each failing sentence into linguistic categories: lexical errors, where the wrong word is chosen, including homophones; contextual errors, where the grammar is inappropriate, such as a tense or agreement mistake; and semantic errors, where the meaning changes. The framework sets out five classification methods of increasing automation and cost. Manual annotation, a person reading each failure and assigning a category, is the most defensible. Dictionary-based analysis detects homophones by comparing pronunciations. Grammar-based analysis uses a parser to detect tense, agreement, and word-order errors. Semantic similarity analysis measures meaning preservation. Language-model classification uses an instructed model to label each error. This project uses manual annotation for the primary classification and the dictionary, grammar, and semantic methods as automated cross-checks, reporting where they agree and disagree. A severity assessment accompanies the classification, grading each failure by how much it changes the meaning, which distinguishes a harmless error from one that would mislead a reader.

Together the three stages give a picture no single metric can. Stage 1 establishes the size of the problem, Stage 2 its shape at the level of sounds, and Stage 3 explains it at the level of words and meaning. The application of this framework to the two fine-tuned models is the subject of Chapter 10.
