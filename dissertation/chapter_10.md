# Chapter 10. Error Pattern Analysis

This chapter applies the three-stage error analysis framework of Chapter 6 to the fine-tuned decoder. It analyses two models, the clean-trained decoder and the noise-augmented decoder, on the same 949-sentence deduplicated validation set with the same beam-search decoding, so that any difference between them is a property of the models rather than of the evaluation, and it closes by answering the fourth research question, whether homophones are the main challenge. The analysis is on the validation set; extending it to the held-out test set is noted as future work.

The clean-trained model gets 873 of the 949 sentences exactly right and fails on 76; the noise-augmented model gets 842 right and fails on 107. These failure counts, 76 and 107, are the samples the deeper analysis works from, so several breakdowns below rest on small counts, flagged where they occur.

## 10.1 Stage 1: conventional evaluation

The two models differ on every conventional metric, and the noise-augmented model is worse across the board, which is the clean-input cost of the robustness it gains.

| Metric | Clean-trained | Noise-trained |
| --- | --- | --- |
| Phoneme error rate | 1.75% | 2.26% |
| Word error rate | 2.09% | 2.92% |
| Character error rate | 0.98% | 1.60% |
| Exact sentence match | 92.0% | 88.7% |

One pattern in these numbers guides the next stage. The degradation is not uniform: the character error rate rises proportionally more than the word error rate, which rises more than the phoneme error rate. As measurement moves from sounds up to characters, the noise-augmented model looks progressively worse relative to the clean one. Stage 1 cannot say why, but it points to the wrong words being spelled further from the target, which Stage 2 examines.

## 10.2 Stage 2: phoneme error patterns

The substitution, insertion, and deletion breakdown explains the Stage 1 pattern. At the word level both models are dominated by substitutions rather than insertions or deletions, so the main failure is choosing the wrong word rather than adding or dropping words. At the character level the noise-augmented model shows a sharp rise in insertions, from 67 to 146, far more than the rise in substitutions or deletions, and this drives its higher character error rate: its extra damage is added characters rather than swapped words.

The confusion matrix records, for each reference sound that was misproduced, which sound the model produced instead. Figure 10.1 shows this for the clean-trained model. Reference sounds run down the rows and predicted sounds across the columns, with vowels grouped in the upper-left block.

![Figure 10.1. Phoneme substitution confusion matrix for the clean-trained model across its failing sentences. Vowels are grouped top-left, consonants bottom-right. Only two cells reach a count of two; every other confusion happens once.](figures/confusion_clean.png)

The matrix for the clean-trained model holds 38 substitutions across 36 distinct cells: only two cells reach a count of two, and every other confusion happens once. Spread this thinly, the model does not confuse one sound for another in a consistent way a targeted rule could correct, so its errors do not repeat. Half of the substitutions, 19 of 38, are a vowel replaced by another vowel, the largest single group, so the model's weak point is vowel quality rather than consonants.

The noise-augmented model shows two changes in the shape of these errors, beyond their higher count. Its confusions repeat more: 14 pairs occur more than once, against only two for the clean model, led by two vowel confusions that each occur three times. Its errors are also more phonetically close: a weighted phoneme error rate, which discounts a substitution between two similar sounds, sits at 0.87 of the unweighted rate for the noise model against 0.93 for the clean model, so its wrong sounds sit nearer the correct ones. Both changes indicate that noise training concentrated the errors onto a recurring set of near-miss vowel confusions.

An analysis by articulatory feature supports this: grouping the substitutions by whether they preserve place, manner, and voicing, the noise-augmented model preserves each feature slightly more often than the clean model, the feature-level view of its errors being more phonetically close.

## 10.3 Stage 3: hierarchical error analysis

Stage 3 classifies the failing sentences into the framework's linguistic categories. The primary classification is by manual annotation, with the dictionary, grammar, and semantic methods as automated cross-checks.

The lexical classification assigns each failure to one dominant category.

| Error type | Clean-trained | Noise-trained |
| --- | --- | --- |
| Other (non-words, proper nouns, number formatting, boundary splits) | 50 (65.8%) | 49 (45.8%) |
| Lexical (wrong real word) | 15 (19.7%) | 35 (32.7%) |
| Homophone | 6 (7.9%) | 11 (10.3%) |
| Contextual (grammatical) | 5 (6.6%) | 10 (9.3%) |
| Semantic (meaning changed) | 0 | 2 (1.9%) |

The dictionary-based check, which labels a substitution a homophone only when the two words share an exact pronunciation, found four such cases for the clean model and six for the noise model, all well-known homophones such as "to" for "too" and "by" for "buy". This strict count is lower than the manual one because the manual reading also catches near-identical sound-alikes the dictionary does not treat as exact matches.

The grammar-based check, which detects tense, agreement, and word-order errors, fired on no clean-model failures and on nine noise-model failures, split between tense, agreement, and inserted function words. Grammar is a minor error source for both models, and their predictions are mostly well-formed.

The semantic analysis uses the sentence-embedding measure, more sensitive to a change in meaning than the token-level measure. On the failing sentences the clean model has a mean sentence similarity of 0.66 and the noise model 0.67, and about a fifth of each model's failures fall below the threshold indicating genuine meaning loss. An earlier token-level measurement had reported that almost no failures lost meaning, an artefact of that measure rewarding word overlap; the sentence-level measure shows that roughly one failure in five does change meaning.

The severity assessment, grading each failure by meaning impact using the sentence-level score, places more than half of each model's failures at substantial meaning change or worse. The two severity distributions are close, matching the semantic finding that their failures are of similar seriousness per case.

## 10.4 Clean versus noise-augmented model

Bringing the stages together shows exactly what noise-augmented training changed. The clearest evidence is in the absolute counts of the lexical classification. The Other category, holding non-words, mangled proper nouns, and formatting quirks, is almost identical between the two models, 50 against 49, so noise training added no garbled output. Every one of the 31 extra failures it introduced landed in the linguistic categories, and 20 of the 31 are wrong real words. The two models fail on the same hard cases, the roughly 50 proper nouns and rare words that neither can handle, and the cost of noise training is more confident wrong real words on clean input. This matches Stage 2, where the noise model's errors were more phonetically close and more real-word-like: the same behaviour appears at the word level as a shift from non-words toward wrong real words.

The semantic analysis refines the cost. Per failure, the two models preserve meaning about equally, and both lose meaning on about a fifth of their failures. The noise model's semantic cost is therefore in the number of failures rather than their seriousness: it produces more failures, so slightly more meaning-lost sentences in absolute terms, but each failure is no worse than the clean model's. This is the honest reading of what the robustness of Chapter 9 costs on clean input.

## 10.5 Are homophones the main challenge?

The fourth research question asks whether homophones are the dominant error type. Across both models they are not. By the strict dictionary definition, homophones are 4 of the clean model's failures and 6 of the noise model's, around 5% in each case; by the broader manual definition, which includes near-identical sound-alikes, they reach about 8% for the clean model and 10% for the noise model. Either way they are a small minority. The dominant categories are wrong real words and the mixed Other category of non-words, rare names, and formatting, none of which a homophone-specific method would address.

This result corroborates the earlier study on this task, which found homophone confusions to be about 1% of the errors of a Flan-T5 decoder and the contextual and phonetic categories to dominate (Hossain et al., under review). Finding the same conclusion with a different model, a decoder-only rather than an encoder-decoder, strengthens the case. The practical consequence is that effort on this task is better directed at the error types that dominate, the wrong-word substitutions and the handling of rare and out-of-vocabulary words, than at homophone-specific handling.
