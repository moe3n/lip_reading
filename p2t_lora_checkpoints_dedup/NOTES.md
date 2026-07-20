# Working notes — staged error analysis (dedup + beam-5)

Scope: error analysis on `p2t_lora_checkpoints_dedup/predictions_beam5.csv` (949 rows, dedup val, beam-5). Companion to `RESULTS_SUMMARY.md`.

## Methodology discipline (load-bearing — read before adding any analysis)

1. **One analysis per step.** No pre-baked pipelines. After each step, in order:
   - what the metric says (the number)
   - what pattern is visible in the *actual failing examples* (read them, do not just read the rate)
   - the hypothesis that pattern supports
   - what the next analysis would be that confirms or kills that hypothesis
   *only then* choose the next step.
2. **Manual intervention is expected.** Before making a judgement call — which phoneme to drill into, which feature mapping to use, how to categorise an ambiguous error — stop and confirm.
3. **No GPU scripts without asking.** Describe the command instead; the dedup predictions are already on disk.
4. **No sandbox-only claims.** Anything reported as a finding must have been measured on these predictions (or the explicit greedy / full twin), not on a simulated run.
5. **Mark low-count cells.** With only 949 val rows the absolute error count is ~80. Any confusion cell with count ≤ 2 should be flagged as low-confidence, not reported as a pattern.

## Code state — what is built, what's unrun, what's missing

**Built and in regular use** (already produces the headline numbers above):
- `src/p2t_lora/evaluation/metrics.py::stratified_evaluate()` — WER / CER / BLEU-4 / Exact Match stratified by homophone mask (Stage 1).

**Built, runs inside `dryrun.py`, but not yet applied to dedup predictions:**
- `src/p2t_lora/evaluation/error_analysis.py`
  - `classify_substitution(ref_word, hyp_word)` — Homophone / Near-homophone / Other
  - `analyze_pair()` — jiwer word alignment + per-substitution classification
  - `error_category_report()` — aggregates across the set, split by homophone mask
- `src/p2t_lora/evaluation/contextual_analysis.py::check_grammar()` — Stage 3 Option 3; only fires on Homophone / Near-homophone subs; only catches closed-class dependency mismatches (`their/there/your/its`).
- `src/p2t_lora/evaluation/extended_metrics.py`
  - `allophonic_error_rate(refs, hyps)` — AER, place / manner / voicing breakdown
  - `weighted_per(refs, hyps, method="heuristic")` — WPER
  - `grammar_error_rate(hyps)` — language_tool_python
  - `semantic_similarity(refs, hyps)` — BERTScore
  - `error_type_breakdown`, `error_type_summary`, `top_confusions`

**Exists but disabled / untrustworthy:**
- `llm_judge.py` — Stage 3 Option 5; off by default, classifications documented as untrustworthy at this scale.

**Has never been done on any of our runs:**
- PER (phoneme error rate) for the model's outputs.
- PER confusion matrix (ref-phoneme → hyp-phoneme counts).
- Per-phoneme drill-down.

## Reference methodology — Mira Fleite (peer student, do NOT anchor on numbers)

Mira did this analysis for a prompting-only ablation on Llama 3.1 8B Instruct, same corpus, same split, **no training**. Her regime is 0% exact match on essentially all rows; ours is ~92% exact match on dedup val. **So her error distribution is not our expected distribution** — order of analyses is the lesson, not the magnitudes.

Her order:
1. SID breakdown (word + char)
2. PER + confusion matrix
3. Top-N phoneme confusions
4. WPER (feature-weighted)
5. AER (place / manner / voicing)
6. Per-phoneme drill-down (she picked `/ah/`)
7. Grammar analysis (language_tool_python)
8. Contextual / semantic (spaCy + BERTScore)
9. Homophone analysis
10. Consolidated priority-ordered summary (Tables 19–20 style)

Her zero-shot headline: WER 143%, PER 73%, `/ah/` worst, WPER ≪ PER (errors were phonetically close), manner 23.4%, grammar 7.7% of errors, homophones 3.91%, BERTScore 0.50.

## Eyeballed observations on dedup beam-5 (leads only — not quantified, not findings)

These are visible-by-eye patterns in `predictions_beam5.csv`. None has been measured. Do not report any of them as a finding until the relevant SID / confusion step confirms it.

| Pattern | Examples (ref → hyp) | Hypothesis it could support |
|---|---|---|
| Number formatting mismatch | `TEN → 10`, `SIX → 6`, `THREE → 3`, `9 11 → 9/11` | Phonetically correct; fails only on orthographic convention. May inflate WER without indicating a real decoding failure. |
| Compound word splitting | `SOUTHWELL → SOUTH WELL`, `ROPES → ROPE S`, `CROSSOVER → CROSS OVER`, `RUNDOWN → RUN DOWN`, `BEFOREHAND → BEFORE HAND`, `DATABASE → DATA BASE`, `TOPLINE → TOP LINE`, `FOREPLAY → FOUR PLAY` | Possibly the single most common error shape. Tests whether the model has internalised closed-form compounds. |
| Proper-noun OOV | `WALDORF ASTORIA → WALLDORF STORY A`, `JENNY ECLAIR → GENIE CLARKE`, `KHAN → CONNIE`, `SYRIA → SURREY`, `GLOUCESTER → GLOSTER`, `LLOYD → LOYD` | Rare / unstressed proper nouns. Phonetically plausible substitutions. |
| Rare / low-frequency words | `SAXIFRAGE → SAXOPHONE FRAGMENT`, `PORTCULLIS → PERKELLIS`, `POLYGAMY → POLIGAMY`, `SQUIRREL → SQUWAREL` | Vocabulary coverage gap, not phoneme recognition gap. |
| Casing artifacts | `no SOUND`, `same TIME` | Output post-processing irregularity. Mechanical, low research value but inflates CER. |

## File-level caveats

- `predictions_beam5.csv` has a literal `\n` inside the `prediction` field of row 4 (`WHEN THERE ISN'T MUCH ELSE IN THE GARDEN` → `WHEN THERE ISN'T MUCH ELSE IN THE GARDE\nN`). Verified with `csv.reader`: 950 rows total (1 header + 949 data), every row has exactly 3 columns. The newline is harmless under CSV parsing — only line-number-derivation off the raw file (`Get-Content`, `readlines`, `wc -l`) is off by one. All our analysis tools (`jiwer`, `error_analysis`) use csv-style parsing, so they see the correct 949-row set.
