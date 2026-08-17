# Report code

The code used in the dissertation *Phoneme-to-Text Conversion for Automated Lip-Reading*,
copied here as a self-contained snapshot. Paths mirror the original repository, so the
`sys.path` insertions that add `src/` still resolve within this folder.

Not included: the raw LRS2 data (licensed, not redistributed), the model checkpoints,
the `src/cpt_decoder/` package (the contrastive method that was not used in the report),
and the exploratory one-off scripts under `p2t_lora_checkpoints_dedup/` (see note at end).

Install: `pip install -r requirements-p2t-lora.txt`.

## Core package: `src/p2t_lora/`

| File | Role |
| --- | --- |
| `model.py` | Loads Llama-3.2-3B in 4-bit and attaches the QLoRA adapters; four-bit safe-`.to` patch. |
| `dryrun.py` | Training driver (QLoRA fine-tune, `TrainingArguments`); full runs set the data size by env var. |
| `data/loader.py`, `data/g2p.py` | LRS2 pair loading, split, deduplication; grapheme-to-phoneme for the phoneme error rate. |
| `augmentation/phoneme_noise.py` | Substitute / delete / insert corruption for noise-augmented training and the robustness probe. |
| `evaluation/metrics.py` | WER, CER, PER, exact match, BLEU; stratified (homophone) reporting. |
| `evaluation/extended_metrics.py`, `error_analysis.py`, `contextual_analysis.py` | Stage-2 and Stage-3 error-analysis support. |
| `evaluation/llm_judge.py` | The fifth error-classification method (implemented, not run in the report). |

## Baselines and evaluation (root scripts)

| File | Report use |
| --- | --- |
| `direct_baseline.py` | GRU no-language-model baseline (Chapters 5, 9); `--official` trains on 45,839 and evaluates on the test set. |
| `eval_test_clean.py` | Held-out test-set evaluation, beam-5, standard and deduplicated (Chapter 9, Tables 9.1, 9.3). |
| `regen_dedup_beam5.py` | Regenerates the deduplicated beam-5 validation figures (Chapter 9). |
| `analyze_zeroshot.py` | Zero-shot baseline analysis (Chapters 5, 9). |
| `analyze_errors.py` | Error extraction over a checkpoint's predictions. |

## Zero-shot baseline: `zero-shot/`

| File | Report use |
| --- | --- |
| `run_baseline.py` | Runs the pretrained model with clean and raw prompt formats. |
| `make_zero_shot_clean_report.py` | Summarises the zero-shot result. |
| `analysis/check_prompt_truncation.py` | Checks prompts are not truncated. |
| `baseline_with_full_corpus/*.py` | Full-corpus zero-shot run: sharding, merge, exact match, re-analysis, results docx. |

## Error analysis and figures: `analysis/`

| File | Report use |
| --- | --- |
| `error_pattern_analysis.py` | Three-stage error-analysis driver (Chapter 10). |
| `step1_sid.py` | Stage 1 substitution/insertion/deletion counts (Chapter 10). |
| `stage3_manual_annotation_clean.py`, `stage3_manual_annotation_noise.py` | Stage-3 manual classification of failures for each model. |
| `stage3_option4_sbert*.py`, `stage3_semantic_noise.py`, `stage3_options234_noise.py` | Stage-3 dictionary/grammar/semantic cross-checks; sentence-embedding semantic scoring. |
| `noise_probe.py`, `_emit_noise_probe_inputs.py`, `make_noise_figures.py` | Robustness corruption probe and Figure 9.1. |
| `make_runs_overview.py` | Cross-run comparison overview. |
| `md_to_docx.py` | Markdown-to-docx renderer used by the dissertation build. |

## Confusion-matrix figure: `figures/`

| File | Report use |
| --- | --- |
| `_phoneme_confusion_fixed.py` | Phoneme-level (whitespace-tokenised) confusion counts for Figure 10.1. |
| `_make_confusion_matrix_figure.py` | Renders the confusion-matrix figure. |

## Dissertation build: `dissertation/`

| File | Role |
| --- | --- |
| `build_dissertation.py` | Builds `dissertation.docx` to the LSBU format from the chapter markdown. |
| `recover_chapters.py` | Regenerates the chapter markdown from the docx if the sources are lost. |

## Note on excluded exploratory scripts

The earlier, iterative lineage of the error analysis lives under `p2t_lora_checkpoints_dedup/`
in the original repository (`step2_classify_failures*.py`, `step3_extended_metrics.py`,
`step4_grammar_semantic.py`, and many `_make_*`, `_reorder_*`, `_inject_*`, `_append_*`
one-off scripts that produced an interim findings document). The reusable modules under
`analysis/` supersede them, so they are not copied here. Pull any of them in if a specific
interim artefact needs to be reproduced.
