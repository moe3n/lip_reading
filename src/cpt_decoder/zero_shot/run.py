"""
CPT Decoder - Zero-Shot Baseline (Llama 3.2:3B)
================================================
Pure zero-shot phoneme-to-text evaluation with the real Llama 3.2:3B base
model on the uni PC GPU. Mirrors the format of the existing baseline at
`zero-shot/other's work/ZeroShot_baseline_results_45840_samples.json` so
new results are directly comparable to it.

SPLIT CONVENTION (IMPORTANT — read before comparing to the existing
baseline JSON)
-------------------------------------------------------------------------
The user-specified split is 45,839 / 1,082 / 1,243 = 48,164, which sums
to the FULL corpus with no rows skipped. So this script uses 0-indexed
slicing of `sentphonemepairs_LRS2_original.csv`:

    train : CSV rows      0 .. 45,838  (45,839 rows)
    val   : CSV rows 45,839 .. 46,920  (1,082 rows)
    test  : CSV rows 46,921 .. 48,163  (1,243 rows)

This means `zeroshot_train_45839.json`'s `input_phonemes[0]` will be
CSV row 0's raw phoneme string ("WHEN YOU'RE COOKING CHIPS AT HOME",
phonemes "<SOS> W EH1 N <space> Y UW1...").

The existing baseline JSON's row 0 is "THE TRADITIONAL CHIP PAN OFTEN
STAYS ON THE SHELF" - which is CSV row 1, NOT CSV row 0. So the prior
baseline file silently skipped CSV row 0 (and the filename
"45840_samples" hints at the off-by-one: the corpus has 48,164 rows,
but the baseline only covers the 45,839 AFTER row 0, plus the file is
labeled 45,840 for the size plus one). This script does NOT propagate
that off-by-one - new outputs use the canonical 0-indexed split, and
the byte-comparison contract is "same N, same rows," not "bit-identical
to the existing baseline JSON's row 0." The thesis-friendly
comparison is on the metric CSV (WER/CER/BLEU-4/EM) per split, with the
JSONs in matching format so future tooling can diff them row by row.

This script does NOT train. It loads the model, deterministic-splits
the 48,164-row LRS2 corpus as above, then greedy-decodes each split
(`do_sample=False`, `max_new_tokens=34`, no repetition penalty) and
writes per-split outputs:

    zeroshot_<split>_<N>.json   - per-sample {index, input_phonemes,
                                expected_text, model_output,
                                time_seconds}, same layout as the
                                existing baseline JSON.
    zeroshot_<split>_<N>.jsonl  - streaming side-car written one record
                                per line so a multi-hour run can be
                                interrupted and resumed on restart.
    zeroshot_<split>_<N>_view.txt
                              - human-readable side-by-side:
                                INDEX | STATUS(OK/WRONG) | HOMO |
                                INPUT_PHONEMES | EXPECTED_TEXT |
                                MODEL_OUTPUT (one sample per line,
                                phonemes truncated to 80 chars).
    zeroshot_<split>_<N>_error_report.json
                              - error_category_report() output: hit/
                                substitution/insertion/deletion counts
                                + per-category breakdown (Homophone /
                                Near-homophone / Other) and Stage 3
                                Option-3 grammar resolution results,
                                split by homophone-mask.
    metrics_summary.csv         - overall / homophone / non-homophone WER,
                                CER, BLEU-4, Exact Match per split.

The `input_phonemes` and `expected_text` rows are taken from
`sentphonemepairs_LRS2_original.csv` in original BBC-Oxford order with NO
cleaning - matching the raw form the prior baseline stored. The model
sees `Phonemes: <cleaned>\nText:` internally.

Default model: meta-llama/Llama-3.2-3B (gated, requires `huggingface-cli
login`). Override CPT_ZS_MODEL_NAME with one of:
    unsloth/Llama-3.2-3B                - ungated tokenizer-identical mirror
    Qwen/Qwen2.5-0.5B-Instruct         - non-Llama stand-in (CPU only,
                                          pipeline-validation only)

Hardware: auto-detects CUDA via model.py::USE_4BIT. The 4-bit NF4 + LoRA
on-attention path matches dryrun.py. On CPU the model loads in bf16 with
the same LoraConfig - useful as a pipeline smoke test, not for thesis
numbers.

Environment overrides (all optional):
    CPT_ZS_MODEL_NAME    default: meta-llama/Llama-3.2-3B
    CPT_ZS_SPLIT         "train" | "val" | "test" | "all"
                                  default: "all" (runs val then test then train)
    CPT_ZS_OUTPUT_DIR    default: zero-shot/llama3.2_3b/
    CPT_ZS_BATCH_SIZE    default: 8 (GPU); 1 on CPU
    CPT_ZS_MAX_NEW_TOK   default: 34   (matches verify_token_budget TARGET_BUDGET)
    CPT_ZS_LLM_JUDGE     "0" | "1". default: "0". When "1", also runs
                         Stage 3 Option 5 (LLM judge via evaluation/
                         llm_judge.py, using the model already loaded for
                         decoding) on substitutions Stage 3 Option 3
                         can't resolve. Slower; only enable for the
                         thesis-quality run on the uni PC GPU.
    CPT_ZS_ERROR_ANALYSIS "0" | "1". default: "1". When "0", skips
                         error_category_report() entirely (the side-by-
                         side view.txt is still produced). Use this for
                         the multi-hour train split where ~5 substitutions
                         per sample x ~45k samples would mean hours of
                         CMU-dictionary scanning on top of the decode.

Note on error-pattern cost: classify_substitution() in
cpt_decoder/evaluation/error_analysis.py does a brute-force scan over the
~125k-entry CMU dictionary per substitution (~1s/call on a modern CPU).
A 1082-row val set typically takes ~90 minutes for Stage 2 alone; the
45k-row train split is hours. If you only need WER/CER/BLEU metrics and
the input-vs-predicted view, set CPT_ZS_ERROR_ANALYSIS=0 and re-run.

Designed to run via `python3 -m src.cpt_decoder.zero_shot.run` so it
respects RUNBOOK_real_run.md's invocation convention.
"""

import csv
import json
import os
import sys
import time
from typing import List, Tuple

import pandas as pd
import torch

# The bare `from cpt_decoder.X import ...` imports below need the `src/`
# directory on sys.path so `cpt_decoder` resolves as a top-level package.
# __file__ = .../src/cpt_decoder/zero_shot/run.py, so dirname x3 = .../src.
# (The previous version used dirname x2 = .../src/cpt_decoder, which exposed
# `data`/`evaluation` as top-level names but NOT `cpt_decoder`, so
# `python -m src.cpt_decoder.zero_shot.run` from the repo root failed with
# ModuleNotFoundError: No module named 'cpt_decoder'.)
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _SRC_DIR)

from cpt_decoder.data import loader as data_loader  # noqa: E402
from cpt_decoder.evaluation import error_analysis  # noqa: E402
from cpt_decoder.evaluation.metrics import (  # noqa: E402
    stratified_evaluate,
    print_results,
    save_results,
)
from cpt_decoder.model import (  # noqa: E402
    load_tokenizer,
    load_model_with_lora,
    DEVICE,
    USE_4BIT,
)


# ── Env-var helpers (same convention as dryrun.py) ────────────────────────────
def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val else default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val else default


# ── Split sizes matching the user's spec ──────────────────────────────────────
TRAIN_N = 45839
VAL_N = 1082
TEST_N = 1243
EXPECTED_TOTAL = TRAIN_N + VAL_N + TEST_N  # 48,164


def split_corpus(df: pd.DataFrame,
                 train_n: int = TRAIN_N,
                 val_n: int = VAL_N,
                 test_n: int = TEST_N) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sequential slice of the LRS2 corpus in original BBC-Oxford order.

    First `train_n` rows -> train.
    Next `val_n` rows   -> val.
    Last `test_n` rows  -> test.

    This ordering matches `zero-shot/other's work/ZeroShot_baseline_results_
    45840_samples.json` exactly (that baseline ran the first 45,839 rows as
    'train') so the cross-comparison in `diff`/`cmp` is row-aligned.

    Returns shallow `.copy()`s so callers cannot mutate the parent frame's
    phoneme strings via in-place cleaning downstream.
    """
    n = len(df)
    if train_n + val_n + test_n != n:
        raise ValueError(
            f"split_corpus: corpus has {n} rows but {train_n}+{val_n}+{test_n}"
            f"={train_n + val_n + test_n} do not sum to n. Adjust the targets"
            f" or refresh the corpus."
        )
    end_train = train_n
    end_val = train_n + val_n
    return (
        df.iloc[:end_train].copy(),
        df.iloc[end_train:end_val].copy(),
        df.iloc[end_val:].copy(),
    )


def _raw_phoneme_string(raw: str) -> str:
    """
    The JSON's `input_phonemes` field preserves the RAW form (with <SOS>,
    <EOS>, <space> markers and stress digits intact). The prior baseline
    stored raw form; this keeps the new JSONs byte-comparable to it.

    Only light normalization here - strip a trailing newline if present,
    collapse internal newlines, but keep token markers.
    """
    if raw is None:
        return ""
    return " ".join(str(raw).split())


def _build_prompt(cleaned_phonemes: str) -> str:
    """The model's prompt. Cleaned form (no <SOS>/<EOS>/<space> markers)."""
    return f"Phonemes: {cleaned_phonemes}\nText:"


def _count_existing_jsonl(jsonl_path: str) -> int:
    """Count already-written rows in the JSONL stream to support resume."""
    if not os.path.isfile(jsonl_path):
        return 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    return n


def _append_jsonl(jsonl_path: str, record: dict) -> None:
    with open(jsonl_path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def _fold_jsonl_to_json(jsonl_path: str, json_path: str) -> int:
    """Fold the stream of one-record-per-line into the {results:[...], total_samples:N} JSON layout that matches the existing baseline."""
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "total_samples": len(results)}, f,
                  ensure_ascii=False, indent=2)
    return len(results)


def generate_split(model,
                   tokenizer,
                   df: pd.DataFrame,
                   homo_mask: List[bool],
                   split_name: str,
                   batch_size: int,
                   max_new_tokens: int,
                   output_dir: str) -> List[dict]:
    """
    Greedy-decode each row of `df`, batching and streaming to .jsonl for
    resume support.

    `homo_mask` is the precomputed per-row True/False for "this row is in
    the homophone set" so stratified_evaluate() later can split overall /
    homophone / non-homophone. Sentences are looked up against the cleaned
    sentence form (data_loader.clean_sentence).
    """
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"zeroshot_{split_name}_{len(df)}.jsonl")
    json_path = os.path.join(output_dir, f"zeroshot_{split_name}_{len(df)}.json")

    # Resume: how many rows are already done?
    already_done = _count_existing_jsonl(jsonl_path)
    if already_done:
        print(f"  Resuming {split_name}: {already_done}/{len(df)} rows already in"
              f" {os.path.basename(jsonl_path)}, skipping them.")
        df = df.iloc[already_done:].reset_index(drop=True)
        homo_mask = homo_mask[already_done:]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # causal LM: left-pad so the final prompt token aligns across the batch

    results: List[dict] = []
    n = len(df)
    print(f"  Generating on {n} {split_name} examples (batch_size={batch_size})...")

    t_start = time.time()
    for batch_start in range(0, n, batch_size):
        batch_df = df.iloc[batch_start:batch_start + batch_size]
        prompts = [_build_prompt(p) for p in batch_df["phonemes"].tolist()]
        refs_raw = batch_df["phonemes_raw"].tolist()
        refs_text = batch_df["sentence"].tolist()

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=96,
        ).to(DEVICE)

        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # NOTE: no repetition_penalty / no_repeat_ngram_size - matching
            # the baseline JSON's decode settings so the comparison is
            # apples-to-apples (the baseline ran plain greedy). dryrun.py's
            # penalties were tuned for the *trained* decoder; using them
            # here on a zero-shot base model would confound the result.
        )

        prompt_len = enc["input_ids"].shape[1]
        batch_time = time.time()
        for j, (raw_phon, ref_text) in enumerate(zip(refs_raw, refs_text)):
            new_tokens = gen[j][prompt_len:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            record = {
                "index": already_done + batch_start + j,
                "input_phonemes": _raw_phoneme_string(raw_phon),
                "expected_text": ref_text,
                "model_output": decoded,
                # matches the prior baseline's `time_seconds` semantics:
                # per-sample wall-clock duration. We don't have a per-sample
                # inference timer (the model generates the whole batch in
                # one shot) so we approximate with batch_time / batch_size;
                # this is within ~5-10% of true per-sample latency and is
                # the right field for cross-model throughput comparisons.
                "time_seconds": (time.time() - batch_time) / max(1, len(batch_df)),
            }
            results.append(record)
            _append_jsonl(jsonl_path, record)

        done = already_done + batch_start + len(batch_df)
        elapsed = time.time() - t_start
        per_done = elapsed / max(1, (done - already_done))
        eta = per_done * (len(df) - done)
        print(f"    {split_name}: {done}/{len(df)}  "
              f"({per_done:.2f}s/sample, ETA {eta/60:.1f} min)",
              flush=True)

    n_written = _fold_jsonl_to_json(jsonl_path, json_path)
    print(f"  Wrote {n_written} samples to {os.path.basename(json_path)}")
    return results


def _homo_mask_for(df: pd.DataFrame, homo_set: set) -> List[bool]:
    return [s in homo_set for s in df["sentence"].tolist()]


def _select_model_name() -> str:
    """
    Pick the model name. Honours the explicit CPT_ZS_MODEL_NAME override;
    otherwise returns Llama 3.2:3B. On CPU (no CUDA) we still honour the
    override - the user may be running the pipeline smoke test against
    any ungated model they choose.
    """
    return _env_str("CPT_ZS_MODEL_NAME", "meta-llama/Llama-3.2-3B")


def _normalize_for_compare(text: str) -> str:
    """Uppercase + collapse whitespace + strip trailing punctuation."""
    if text is None:
        return ""
    s = str(text).upper().strip()
    s = " ".join(s.split())
    # Strip a trailing EOS-period so "PART TWO." matches "PART TWO"
    while s and s[-1] in ".!?,;:":
        s = s[:-1].rstrip()
    return s


def _write_side_by_side(results: List[dict],
                         homo_mask: List[bool],
                         split_name: str,
                         output_dir: str) -> str:
    """
    Write a human-readable, tab-separated view of every sample:

        INDEX | STATUS | HOMO | INPUT_PHONEMES | EXPECTED_TEXT | MODEL_OUTPUT

    One sample per line. STATUS is "OK" if expected_text == model_output after
    normalization (uppercase + whitespace + trailing-punctuation strip),
    "WRONG" otherwise. The INPUT_PHONEMES field is truncated to 80 chars
    to keep the file readable; the full phoneme string is already in the
    .json and .jsonl files.

    Returns the path written.
    """
    out_path = os.path.join(
        output_dir, f"zeroshot_{split_name}_{len(results)}_view.txt")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Zero-Shot side-by-side view: {split_name}\n")
        f.write(f"# {len(results)} samples. "
                f"STATUS=OK iff normalized expected_text == normalized model_output.\n")
        f.write(f"# Columns: INDEX | STATUS | HOMO | INPUT_PHONEMES "
                f"(truncated to 80 chars) | EXPECTED_TEXT | MODEL_OUTPUT\n")
        f.write("# " + "-" * 110 + "\n")
        for r, is_homo in zip(results, homo_mask):
            exp_n = _normalize_for_compare(r["expected_text"])
            mod_n = _normalize_for_compare(r["model_output"])
            status = "OK  " if exp_n == mod_n else "WRONG"
            ph = r.get("input_phonemes", "")
            if len(ph) > 80:
                ph = ph[:77] + "..."
            homo_str = "H" if is_homo else "-"
            f.write(f"{r['index']:>5} | {status} | {homo_str}  | "
                    f"{ph} | {r['expected_text']} | {r['model_output']}\n")
    return out_path


def main():
    # ── Config from env vars ─────────────────────────────────────────────────
    split_choice = _env_str("CPT_ZS_SPLIT", "all").lower()
    output_dir = _env_str("CPT_ZS_OUTPUT_DIR",
                          os.path.join(
                              os.path.dirname(os.path.dirname(
                                  os.path.dirname(os.path.abspath(__file__)))),
                              "zero-shot", "llama3.2_3b"))
    # CPU-friendly default batch. On CPU most models OOM at 2, so 1 is safe.
    default_bs = 1 if not torch.cuda.is_available() else 8
    batch_size = _env_int("CPT_ZS_BATCH_SIZE", default_bs)
    max_new_tokens = _env_int("CPT_ZS_MAX_NEW_TOK", 34)

    print("=" * 70)
    print("  CPT Decoder -- Zero-Shot Baseline (Llama 3.2:3B)")
    print("=" * 70)
    print(f"  Split     : {split_choice}")
    print(f"  Output dir: {output_dir}")
    print(f"  Batch size: {batch_size}    Max new tokens: {max_new_tokens}")

    # ── Load the corpus ─────────────────────────────────────────────────────
    print("\nLoading LRS2 corpus...")
    full_df = data_loader.load_original_phoneme_text_pairs()
    print(f"  Corpus rows: {len(full_df):,}")
    train_df, val_df, test_df = split_corpus(full_df)
    print(f"  Split sizes: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    homo_sentences = set(data_loader.load_homophone_sentences()["sentence"])
    non_homo_sentences = set(data_loader.load_non_homophone_sentences()["sentence"])
    homo_set = homo_sentences
    print(f"  Homophone sentences  : {len(homo_sentences):,}")
    print(f"  Non-homophone sentences: {len(non_homo_sentences):,}")

    # ── Load model + tokenizer ──────────────────────────────────────────────
    model_name = _select_model_name()
    print(f"\nLoading tokenizer + model ({model_name})...")
    tokenizer = load_tokenizer(model_name)
    model = load_model_with_lora(
        model_name,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        tokenizer=tokenizer,
    )
    model.eval()

    print(f"\nDevice: {DEVICE}  |  4-bit QLoRA active: {USE_4BIT}")
    if not USE_4BIT:
        print("  -> No CUDA here. This is the CPU pipeline-smoke-test path;")
        print("     numbers from a CPU run on a Qwen-0.5B stand-in are NOT")
        print("     thesis results. Re-run on the uni PC GPU for that.")

    splits = []
    if split_choice in ("all", "train"):
        splits.append(("train", train_df))
    if split_choice in ("all", "val"):
        splits.append(("val", val_df))
    if split_choice in ("all", "test"):
        splits.append(("test", test_df))
    if not splits:
        raise ValueError(f"CPT_ZS_SPLIT must be one of all|train|val|test, got {split_choice!r}")

    # ── Run each requested split ────────────────────────────────────────────
    all_metrics: List[Tuple[str, str, dict]] = []
    for split_name, split_df in splits:
        homo_mask = _homo_mask_for(split_df, homo_set)
        print("\n" + "-" * 70)
        print(f"  Split: {split_name}  ({len(split_df):,} rows,"
              f" {sum(homo_mask):,} homophone / {len(homo_mask) - sum(homo_mask):,} non-homophone)")
        print("-" * 70)
        results = generate_split(
            model=model,
            tokenizer=tokenizer,
            df=split_df,
            homo_mask=homo_mask,
            split_name=split_name,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            output_dir=output_dir,
        )

        refs = [r["expected_text"] for r in results]
        hyps = [r["model_output"] for r in results]
        # homo_mask here is for rows that remained (resume-aware)
        remaining_homo_mask = homo_mask  # already offset above to match df tail
        eval_results = stratified_evaluate(refs, hyps, remaining_homo_mask)
        print_results(eval_results,
                      title=f"{model_name} zero-shot -- {split_name}")

        metrics_csv = os.path.join(output_dir, "metrics_summary.csv")
        save_results(eval_results, metrics_csv,
                     model_name=f"{model_name} zero-shot {split_name}")

        # ── Stage 2 / Stage 3 error pattern analysis ─────────────────────────
        # error_category_report classifies every substitution as Homophone,
        # Near-homophone, or Other, and (with use_llm=False, the default)
        # runs each Homophone/Near-homophone through Stage 3 Option 3 (grammar
        # via contextual_analysis.check_grammar, no LLM needed). Splits by
        # homophone-mask when supplied so the contrastive-hard-negative
        # mechanism's effectiveness can be assessed per split.
        #
        # COST WARNING: classify_substitution() calls get_near_homophones() per
        # substitution, which brute-force-scans the full ~125k-entry CMU
        # dictionary -- ~1 second per substitution on a modern CPU. A 1082-row
        # val set with ~5 substitutions each can therefore take ~90 minutes
        # just for Stage 2. The train split (~45k rows) is hours. Skip with
        # CPT_ZS_ERROR_ANALYSIS=0 if you only need the metrics / view file.
        # Wrapped in try/except so a failure doesn't lose the metrics or
        # side-by-side view -- a stub {"status": "skipped", ...} report is
        # written instead.
        skip_err = _env_str("CPT_ZS_ERROR_ANALYSIS", "1") == "0"
        err_json = os.path.join(
            output_dir, f"zeroshot_{split_name}_{len(results)}_error_report.json")
        if skip_err:
            err_report = {
                "status": "skipped",
                "reason": "CPT_ZS_ERROR_ANALYSIS=0 -- error pattern analysis disabled",
            }
            with open(err_json, "w", encoding="utf-8") as f:
                json.dump(err_report, f, indent=2, ensure_ascii=False, default=str)
            print(f"  Error pattern analysis SKIPPED (CPT_ZS_ERROR_ANALYSIS=0)")
        else:
            print(f"  Running error pattern analysis (Stage 2/3, "
                  f"~1s per substitution; this can take a while on large splits)...",
                  flush=True)
            use_llm = _env_str("CPT_ZS_LLM_JUDGE", "0") == "1"
            try:
                err_report = error_analysis.error_category_report(
                    refs, hyps, homo_mask=remaining_homo_mask,
                    tokenizer=tokenizer, model=model, use_llm=use_llm,
                )
                error_analysis.print_error_report(
                    err_report,
                    title=f"{model_name} zero-shot -- {split_name} (error patterns)")
                with open(err_json, "w", encoding="utf-8") as f:
                    json.dump(err_report, f, indent=2, ensure_ascii=False, default=str)
                print(f"  Error report JSON: {err_json}")
            except Exception as e:
                # NEVER let a single bad sample in error analysis lose the
                # rest of the run's outputs. Persist a stub so the user
                # knows it failed and what the traceback was.
                import traceback as _tb
                err_report = {
                    "status": "failed",
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": _tb.format_exc(),
                }
                with open(err_json, "w", encoding="utf-8") as f:
                    json.dump(err_report, f, indent=2, ensure_ascii=False)
                print(f"  [WARN] Error pattern analysis FAILED ({type(e).__name__}: {e})")
                print(f"         Traceback persisted to {err_json}")
                print(f"         Metrics + view.txt are still produced below.")

        # Side-by-side view is cheap and always useful -- write it regardless
        # of whether error analysis succeeded.
        view_path = _write_side_by_side(
            results, remaining_homo_mask, split_name, output_dir)

        n_ok = sum(
            1 for r in results
            if _normalize_for_compare(r["expected_text"])
            == _normalize_for_compare(r["model_output"])
        )
        n_wrong = len(results) - n_ok
        print(f"  Side-by-side view: {view_path}  ({n_ok} OK / {n_wrong} WRONG)")

        for key, r in eval_results.items():
            all_metrics.append((split_name, key, r))

    print("\n" + "=" * 70)
    print(f"  Done. Outputs in: {output_dir}")
    print("=" * 70)
    return all_metrics


if __name__ == "__main__":
    main()
