# Graph Report - .  (2026-07-19)

## Corpus Check
- Large corpus: 127 files · ~1,239,811 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 973 nodes · 1101 edges · 74 communities (67 shown, 7 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 49 edges (avg confidence: 0.7)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Thesis Walkthrough (model + decoding)
- LoRA Model Card (epoch_1)
- LoRA Model Card (epoch_2)
- LoRA Model Card (epoch_3)
- LoRA Model Card (dedup)
- LoRA Model Card (dedup epoch_1)
- LoRA Model Card (dedup epoch_2)
- LoRA Model Card (dedup epoch_3)
- LoRA Model Card (full)
- LoRA Model Card (full epoch_1)
- LoRA Model Card (full epoch_2)
- Hard Negatives (cpt_decoder)
- Hard Negatives (p2t_lora)
- Direct Baseline (seq2seq)
- Extended Metrics (panphon)
- Token Lengths (cpt_decoder)
- Token Lengths (p2t_lora)
- Prompt Truncation & bnb dtype
- Notebooks README & smoke-test
- Core Metrics (cpt_decoder)
- Zero-Shot Report §1 (Configuration)
- Core Metrics (p2t_lora)
- Analyze Zero-shot
- Supervisor Q&A & Per-epoch
- Weekly Report §1-§6
- Dryrun pipeline (cpt_decoder)
- Dryrun pipeline (p2t_lora)
- Dryrun dataframes & error categories
- Token budget verify & clean phonemes
- Error analysis (p2t_lora)
- Zero-shot clean report (docx)
- Progress briefing (meeting)
- Error analysis (cpt_decoder)
- Weekly Update §1-§3
- Dryrun dataframes (small)
- LLM Judge (cpt_decoder)
- Model load & dtype (cpt_decoder)
- LLM Judge (p2t_lora)
- Model load & dtype (p2t_lora)
- Error report PNG analyzer
- Dedup Report §1-§4
- Verify Run (deletion/leakage)
- Weekly report docx (make_v2)
- Contextual grammar (cpt_decoder)
- Contextual grammar (p2t_lora)
- Results Summary & Key claims
- CPTDataset class (cpt_decoder)
- CPTDataset class (p2t_lora)
- Notebook rebuild helpers
- Lora predictions analyzer
- MIRA recompute script
- Test clean evaluator
- Dedup beam5 regenerator
- Notebook source fixer
- Accelerate verifier

## God Nodes (most connected - your core abstractions)
1. `Model Card for Model ID` - 15 edges
2. `Model Card for Model ID` - 15 edges
3. `Model Card for Model ID` - 15 edges
4. `Model Card for Model ID` - 15 edges
5. `Model Card for Model ID` - 15 edges
6. `Model Card for Model ID` - 15 edges
7. `Model Card for Model ID` - 15 edges
8. `Model Card for Model ID` - 15 edges
9. `Model Card for Model ID` - 15 edges
10. `Model Card for Model ID` - 15 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `plot_error_report()`  [INFERRED]
  analyze_errors.py → src/p2t_lora/evaluation/error_analysis.py
- `main()` --calls--> `load_tokenizer()`  [INFERRED]
  src/cpt_decoder/dryrun.py → src/cpt_decoder/model.py
- `main()` --calls--> `print_error_report()`  [INFERRED]
  src/p2t_lora/dryrun.py → src/p2t_lora/evaluation/error_analysis.py
- `main()` --calls--> `load_tokenizer()`  [INFERRED]
  src/p2t_lora/dryrun.py → src/p2t_lora/model.py
- `classify_substitution()` --calls--> `get_homophones()`  [INFERRED]
  src/cpt_decoder/evaluation/error_analysis.py → src/cpt_decoder/augmentation/hard_negatives.py

## Import Cycles
- None detected.

## Communities (74 total, 7 thin omitted)

### Community 0 - "Thesis Walkthrough (model + decoding)"
Cohesion: 0.05
Nodes (40): 1. How generation works during evaluation, 1. The base model, 1. What `cpt_forward()` does, 1. Where the raw data comes from, 2. 4-bit quantization (NF4), and why it's needed, 2. Gradient accumulation and why it's used, 2. How phonemes are cleaned — and the `<space>` bug, 2. The metrics and what they mean (`metrics.py`) (+32 more)

### Community 1 - "LoRA Model Card (epoch_1)"
Cohesion: 0.05
Nodes (37): Bias, Risks, and Limitations, Citation [optional], Compute Infrastructure, Direct Use, Downstream Use [optional], Environmental Impact, Evaluation, Factors (+29 more)

### Community 2 - "LoRA Model Card (epoch_2)"
Cohesion: 0.05
Nodes (37): Bias, Risks, and Limitations, Citation [optional], Compute Infrastructure, Direct Use, Downstream Use [optional], Environmental Impact, Evaluation, Factors (+29 more)

### Community 3 - "LoRA Model Card (epoch_3)"
Cohesion: 0.05
Nodes (37): Bias, Risks, and Limitations, Citation [optional], Compute Infrastructure, Direct Use, Downstream Use [optional], Environmental Impact, Evaluation, Factors (+29 more)

### Community 4 - "LoRA Model Card (dedup)"
Cohesion: 0.05
Nodes (37): Bias, Risks, and Limitations, Citation [optional], Compute Infrastructure, Direct Use, Downstream Use [optional], Environmental Impact, Evaluation, Factors (+29 more)

### Community 5 - "LoRA Model Card (dedup epoch_1)"
Cohesion: 0.05
Nodes (37): Bias, Risks, and Limitations, Citation [optional], Compute Infrastructure, Direct Use, Downstream Use [optional], Environmental Impact, Evaluation, Factors (+29 more)

### Community 6 - "LoRA Model Card (dedup epoch_2)"
Cohesion: 0.05
Nodes (37): Bias, Risks, and Limitations, Citation [optional], Compute Infrastructure, Direct Use, Downstream Use [optional], Environmental Impact, Evaluation, Factors (+29 more)

### Community 7 - "LoRA Model Card (dedup epoch_3)"
Cohesion: 0.05
Nodes (37): Bias, Risks, and Limitations, Citation [optional], Compute Infrastructure, Direct Use, Downstream Use [optional], Environmental Impact, Evaluation, Factors (+29 more)

### Community 8 - "LoRA Model Card (full)"
Cohesion: 0.05
Nodes (37): Bias, Risks, and Limitations, Citation [optional], Compute Infrastructure, Direct Use, Downstream Use [optional], Environmental Impact, Evaluation, Factors (+29 more)

### Community 9 - "LoRA Model Card (full epoch_1)"
Cohesion: 0.05
Nodes (37): Bias, Risks, and Limitations, Citation [optional], Compute Infrastructure, Direct Use, Downstream Use [optional], Environmental Impact, Evaluation, Factors (+29 more)

### Community 10 - "LoRA Model Card (full epoch_2)"
Cohesion: 0.05
Nodes (37): Bias, Risks, and Limitations, Citation [optional], Compute Infrastructure, Direct Use, Downstream Use [optional], Environmental Impact, Evaluation, Factors (+29 more)

### Community 11 - "Hard Negatives (cpt_decoder)"
Cohesion: 0.08
Nodes (30): build_homophone_lookup_table(), build_phoneme_to_words_index(), generate_hard_negatives(), get_homophones(), get_near_homophones(), CPT Decoder — Hard Negative Generator =======================================, Generate hard negative sentences for a given input.      For each word in the, Build a complete lookup table of all homophones in the CMU Dict.     Returns {w (+22 more)

### Community 12 - "Hard Negatives (p2t_lora)"
Cohesion: 0.08
Nodes (30): build_homophone_lookup_table(), build_phoneme_to_words_index(), generate_hard_negatives(), get_homophones(), get_near_homophones(), P2T LoRA Decoder — Hard Negative Generator =====================================, Generate hard negative sentences for a given input.      For each word in the se, Build a complete lookup table of all homophones in the CMU Dict.     Returns {wo (+22 more)

### Community 13 - "Direct Baseline (seq2seq)"
Cohesion: 0.11
Nodes (21): build_vocabs(), clean_phonemes(), clean_text(), collate(), Decoder, detok(), edit_distance(), encode_phonemes() (+13 more)

### Community 14 - "Extended Metrics (panphon)"
Cohesion: 0.09
Nodes (29): allophonic_error_rate(), _dominant_feature(), error_type_breakdown(), error_type_summary(), grammar_error_rate(), _heuristic_wper(), _panphon_wper(), _phoneme_substitutions() (+21 more)

### Community 15 - "Token Lengths (cpt_decoder)"
Cohesion: 0.12
Nodes (25): main(), check_token_lengths.py  Standalone diagnostic: tokenizes the FULL LRS2 corpus, The proposed fix: also remove the literal '<space>' marker token., strip_space_marker(), summarize(), clean_phoneme_seq(), clean_sentence(), dataset_summary() (+17 more)

### Community 16 - "Token Lengths (p2t_lora)"
Cohesion: 0.12
Nodes (25): main(), check_token_lengths.py  Standalone diagnostic: tokenizes the FULL LRS2 corpus (4, The proposed fix: also remove the literal '<space>' marker token., strip_space_marker(), summarize(), clean_phoneme_seq(), clean_sentence(), dataset_summary() (+17 more)

### Community 17 - "Prompt Truncation & bnb dtype"
Cohesion: 0.13
Nodes (21): Prompt token-length / truncation-budget check for run_baseline.py.  Standalone d, bnb_compute_dtype(), build_prompt(), extended(), extract_answer(), load_model(), main(), norm() (+13 more)

### Community 18 - "Notebooks README & smoke-test"
Cohesion: 0.10
Nodes (18): Files, Google Drive layout (everything Colab writes), `notebooks/` — Colab entry point, Refreshing the LRS2 CSVs, Smoke-testing the notebook (no GPU, no gated token), Using the trained adapter back on the uni PC, lip_reading, 0. What changed in the code, and why (+10 more)

### Community 19 - "Core Metrics (cpt_decoder)"
Cohesion: 0.15
Nodes (19): bleu4_score(), character_error_rate(), evaluate(), exact_match(), homophone_disambiguation_rate(), normalise(), CPT Decoder — Evaluation Metrics ================================== Computes W, Compute all metrics for a set of reference/hypothesis pairs.     Returns a dict (+11 more)

### Community 20 - "Zero-Shot Report §1 (Configuration)"
Cohesion: 0.11
Nodes (17): 1.1 Model, 1.2 Inference parameters, 1.3 Data, 1.4 Post-processing, 1.5 Hardware / runtime, 1.6 Metrics computed, 1. Configuration, 2.1 Core metrics (+9 more)

### Community 21 - "Core Metrics (p2t_lora)"
Cohesion: 0.18
Nodes (17): bleu4_score(), character_error_rate(), evaluate(), exact_match(), homophone_disambiguation_rate(), normalise(), P2T LoRA Decoder — Evaluation Metrics ================================== Compute, Compute all metrics for a set of reference/hypothesis pairs.     Returns a dict (+9 more)

### Community 22 - "Analyze Zero-shot"
Cohesion: 0.24
Nodes (13): _analyze_chunk(), extended(), main(), norm(), parallel_error_report(), parse_filename(), phoneme_error_rate(), Offline analyzer for zero-shot baseline predictions. ========================== (+5 more)

### Community 23 - "Supervisor Q&A & Per-epoch"
Cohesion: 0.14
Nodes (13): Draft email reply, Per-epoch loss (no overfitting turn), Q1. Which split did we evaluate on: validation or test?, Q2. Word boundaries: did we keep the `<space>` token, or substitute a marker like `|`?, Q3. Training configuration, Recommendation for the thesis, Reconciling the counts (1,082 vs the expected 1,079), Response to supervisor questions (12 July 2026) (+5 more)

### Community 24 - "Weekly Report §1-§6"
Cohesion: 0.14
Nodes (13): 1. Row-level split integrity, 2. Metric recomputation with independent code, 3. Output integrity, 4. Exact-duplicate leakage between train and validation (the main finding), 5. Near-duplicate leakage, 6. Overfitting check from the training curves, 7. Qualitative error inspection, 8. Plausibility of the magnitude (+5 more)

### Community 25 - "Dryrun pipeline (cpt_decoder)"
Cohesion: 0.19
Nodes (8): contrastive_loss(), cpt_forward(), mean_pool(), Tensor, CPT Decoder — Llama 3.2:3B + QLoRA Dry Run (Phase 1 architecture port) ========, Same pooling function as the Flan-T5 prototype — architecture-agnostic., Unchanged from the Flan-T5 prototype — margin-based cosine hinge loss., Causal-LM version of the Flan-T5 prototype's cpt_forward.      Differences fro

### Community 26 - "Dryrun pipeline (p2t_lora)"
Cohesion: 0.20
Nodes (6): contrastive_loss(), mean_pool(), Tensor, P2T LoRA Decoder — Llama 3.2:3B + QLoRA Dry Run (Phase 1 architecture port) ====, Same pooling function as the Flan-T5 prototype — architecture-agnostic., Unchanged from the Flan-T5 prototype — margin-based cosine hinge loss.

### Community 27 - "Dryrun dataframes & error categories"
Cohesion: 0.18
Nodes (11): build_dryrun_dataframes(), cpt_forward(), main(), Causal-LM version of the Flan-T5 prototype's cpt_forward.      Contrastive branc, Uses the REAL LRS2 phoneme transcriptions (sentphonemepairs_LRS2_original.csv, error_category_report(), Run analyze_pair() across an entire evaluation set and aggregate.      Args:, print_results() (+3 more)

### Community 28 - "Token budget verify & clean phonemes"
Cohesion: 0.31
Nodes (9): ndarray, clean_phoneme_seq_current(), clean_phoneme_seq_fixed(), clean_sentence(), main(), Verify the token-length budget for the CPT Decoder.  Runs the full 48,164-row, Current cleaning in data/loader.py — leaves <space> intact., Proposed fix — also strips the <space> word-boundary marker. (+1 more)

### Community 29 - "Error analysis (p2t_lora)"
Cohesion: 0.27
Nodes (9): analyze_pair(), classify_substitution(), normalise(), print_error_report(), P2T LoRA Decoder — Error Pattern Analysis (Stage 2 + Stage 3 Options 2/3/5) ====, Classify a single substitution (ref_word -> hyp_word) as one of:         "Homoph, Stage 3 of the P2T framework: given a substitution already classified     by Sta, Run jiwer's word-level alignment on a single (reference, hypothesis) pair,     c (+1 more)

### Community 30 - "Zero-shot clean report (docx)"
Cohesion: 0.33
Nodes (9): add_bullets(), add_heading(), add_para(), add_table(), build(), Generate REPORT_zero_shot_clean_train.docx — concise (≤ 2pp) report on the zero, Remove the default page-break-before that python-docx puts on H1., set_calibri() (+1 more)

### Community 31 - "Progress briefing (meeting)"
Cohesion: 0.22
Nodes (8): Anticipated questions and responses, Central finding, Limitations, Next steps, Phoneme → Text Decoder: Progress Briefing, Results (evaluation set, n = 9,633), Summary, Work completed

### Community 32 - "Error analysis (cpt_decoder)"
Cohesion: 0.31
Nodes (8): analyze_pair(), classify_substitution(), normalise(), CPT Decoder — Error Pattern Analysis (Stage 2 + Stage 3 Options 2/3/5) ========, Classify a single substitution (ref_word -> hyp_word) as one of:         "Homop, Stage 3 of the P2T framework: given a substitution already classified     by St, Run jiwer's word-level alignment on a single (reference, hypothesis) pair,, resolve_substitution()

### Community 33 - "Weekly Update §1-§3"
Cohesion: 0.25
Nodes (7): 1. LoRA fine-tuning results, 2. Zero-shot results and analysis, 3. Detailed error pattern analysis, Limitations, Still to come, Summary, Weekly update: LoRA fine-tuning vs. zero-shot prompting

### Community 34 - "Dryrun dataframes (small)"
Cohesion: 0.25
Nodes (8): build_dryrun_dataframes(), main(), Small stratified sample using the REAL LRS2 phoneme transcriptions     (sentpho, error_category_report(), print_error_report(), Run analyze_pair() across an entire evaluation set and aggregate.      Args:, print_results(), Print a formatted evaluation table.

### Community 35 - "LLM Judge (cpt_decoder)"
Cohesion: 0.32
Nodes (7): _build_prompt(), classify_error(), _parse_response(), CPT Decoder — Stage 3 Option 5: LLM-Based Error Classification ================, Builds the user-turn text. Verbatim from the PDF's Option 5 prompt     (Ground, Extract Category/Subcategory/Explanation from the model's free-text     respons, Run the Option 5 LLM-judge prompt for one (ground_truth, prediction)     senten

### Community 36 - "Model load & dtype (cpt_decoder)"
Cohesion: 0.25
Nodes (7): load_model_with_lora(), load_tokenizer(), dtype, CPT Decoder — Model Loading (Llama 3.2:3B + QLoRA) ============================, Load a decoder-only causal LM with QLoRA adapters.      On CUDA: loads in 4-bi, Added 25 Jun 2026, ahead of the first real run on the uni PC's GPUs.      bflo, _select_4bit_compute_dtype()

### Community 37 - "LLM Judge (p2t_lora)"
Cohesion: 0.32
Nodes (7): _build_prompt(), classify_error(), _parse_response(), P2T LoRA Decoder — Stage 3 Option 5: LLM-Based Error Classification ============, Builds the user-turn text. Verbatim from the PDF's Option 5 prompt     (Ground T, Extract Category/Subcategory/Explanation from the model's free-text     response, Run the Option 5 LLM-judge prompt for one (ground_truth, prediction)     sentenc

### Community 38 - "Model load & dtype (p2t_lora)"
Cohesion: 0.25
Nodes (7): load_model_with_lora(), load_tokenizer(), dtype, P2T LoRA Decoder — Model Loading (Llama 3.2:3B + QLoRA) ========================, Load a decoder-only causal LM with QLoRA adapters.      On CUDA: loads in 4-bit, Added 25 Jun 2026, ahead of the first real run on the uni PC's GPUs.      bfloat, _select_4bit_compute_dtype()

### Community 39 - "Error report PNG analyzer"
Cohesion: 0.33
Nodes (6): bnb_compute_dtype(), main(), dtype, Stage 3 Option 5 (LLM judge) error analysis — runs the full error-pattern report, plot_error_report(), Bar chart of substitution error categories (Homophone / Near-homophone     / Oth

### Community 40 - "Dedup Report §1-§4"
Cohesion: 0.29
Nodes (6): 1. Run configuration, 2. Results and comparison, 3. Cause of the gap, 4. Next step, Deduplicated validation run and decoding comparison, Summary

### Community 41 - "Verify Run (deletion/leakage)"
Cohesion: 0.38
Nodes (5): deletion_variants(), main(), norm(), Leakage / hygiene audit for a fine-tuned run's validation predictions.  Read-onl, All strings formed by deleting exactly one word — the hashing trick for     find

### Community 42 - "Weekly report docx (make_v2)"
Cohesion: 0.48
Nodes (6): add_bullets(), add_heading(), add_para(), add_table(), main(), Generate v2.docx — weekly supervisor update covering today's experiments.  Sty

### Community 43 - "Contextual grammar (cpt_decoder)"
Cohesion: 0.38
Nodes (6): check_grammar(), _get_nlp(), CPT Decoder — Stage 3 Option 3: Grammar-Based Contextual Analysis =============, Return the (start_char, end_char) span of the word_idx-th     whitespace-split, Check whether `hyp_word` at whitespace-index `hyp_word_idx` in     `original_hy, _word_char_span()

### Community 44 - "Contextual grammar (p2t_lora)"
Cohesion: 0.38
Nodes (6): check_grammar(), _get_nlp(), P2T LoRA Decoder — Stage 3 Option 3: Grammar-Based Contextual Analysis =========, Return the (start_char, end_char) span of the word_idx-th     whitespace-split w, Check whether `hyp_word` at whitespace-index `hyp_word_idx` in     `original_hyp, _word_char_span()

### Community 45 - "Results Summary & Key claims"
Cohesion: 0.33
Nodes (5): Dataset, Honest caveats to state alongside the numbers, Key settings, P2T Decoder — Results Summary (for supervisor), Results (eval set, n = 9,633)

### Community 46 - "CPTDataset class (cpt_decoder)"
Cohesion: 0.33
Nodes (3): CPTDataset, Dataset, Causal-LM version of the Flan-T5 prototype's CPTDataset.      Each item:

### Community 47 - "CPTDataset class (p2t_lora)"
Cohesion: 0.40
Nodes (3): CPTDataset, Dataset, Causal-LM version of the Flan-T5 prototype's CPTDataset.      Each item:

## Knowledge Gaps
- **353 isolated node(s):** `lip_reading`, `Dataset`, `Results (eval set, n = 9,633)`, `Key settings`, `Honest caveats to state alongside the numbers` (+348 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `classify_substitution()` connect `Error analysis (p2t_lora)` to `Hard Negatives (p2t_lora)`, `Extended Metrics (panphon)`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **Why does `main()` connect `Dryrun dataframes & error categories` to `Model load & dtype (p2t_lora)`, `Error report PNG analyzer`, `CPTDataset class (cpt_decoder)`, `Core Metrics (p2t_lora)`, `Dryrun pipeline (p2t_lora)`, `Error analysis (p2t_lora)`?**
  _High betweenness centrality (0.008) - this node is a cross-community bridge._
- **Why does `get_near_homophones()` connect `Hard Negatives (p2t_lora)` to `Error analysis (p2t_lora)`?**
  _High betweenness centrality (0.006) - this node is a cross-community bridge._
- **What connects `lip_reading`, `Dataset`, `Results (eval set, n = 9,633)` to the rest of the system?**
  _353 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Thesis Walkthrough (model + decoding)` be split into smaller, more focused modules?**
  _Cohesion score 0.04878048780487805 - nodes in this community are weakly interconnected._
- **Should `LoRA Model Card (epoch_1)` be split into smaller, more focused modules?**
  _Cohesion score 0.05263157894736842 - nodes in this community are weakly interconnected._
- **Should `LoRA Model Card (epoch_2)` be split into smaller, more focused modules?**
  _Cohesion score 0.05263157894736842 - nodes in this community are weakly interconnected._