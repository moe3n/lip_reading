"""
Zero-Shot Baseline (ISOLATED) — plain Llama-3.2-3B, phoneme -> text
===================================================================
A self-contained zero-shot baseline for the P2T task. This is deliberately
DECOUPLED from the CPT decoder pipeline: it does NOT use the LoRA/QLoRA model
wrapper (`load_model_with_lora`), the contrastive tooling, or the package's
data/metrics modules. It loads the *plain* base model and greedy-decodes, so
the number it produces is an honest zero-shot floor to compare the fine-tuned
CPT decoder against.

The ONE thing it reuses from the package is the error-pattern analysis
(`cpt_decoder.evaluation.error_analysis`), so the Stage-2/3 homophone /
grammar breakdown is identical across baseline and fine-tuned runs and stays
directly comparable. Set ZS_ERROR_ANALYSIS=0 to skip it (much faster on the
45k train split).

Run it as a plain script (no `-m`, no package import games):

    python zero_shot_baseline.py

Everything else is configured through env vars (all optional):

    ZS_MODEL           default: meta-llama/Llama-3.2-3B  (gated; `huggingface-cli login`,
                                 or use unsloth/Llama-3.2-3B ungated mirror)
    ZS_SPLIT           "train" | "val" | "test" | "all"   default: "all"
    ZS_OUTPUT_DIR      default: zero-shot/baseline
    ZS_BATCH_SIZE      default: 8 on CUDA, 1 on CPU
    ZS_MAX_NEW_TOK     default: 34
    ZS_ERROR_ANALYSIS  "0" | "1"   default: "1"

Outputs, per split, in ZS_OUTPUT_DIR:
    zeroshot_<split>_<N>.json / .jsonl        per-sample records (jsonl streams
                                              for resume; kill & rerun continues)
    zeroshot_<split>_<N>_view.txt             INPUT phonemes | PREDICTED | TARGET
    metrics_summary.csv                       WER / CER / BLEU-4 / ExactMatch,
                                              overall + homophone + non-homophone
    zeroshot_<split>_<N>_error_report.json    substitution error patterns

Decoding is plain greedy (do_sample=False, no repetition_penalty /
no_repeat_ngram_size) to stay apples-to-apples with the existing baseline JSON.

Split convention (matches the CPT pipeline exactly): sequential slice of the
48,164-row corpus in original BBC-Oxford order — train 0..45,838, val
45,839..46,920, test 46,921..48,163.
"""

import csv
import datetime
import json
import os
import re
import sys
import time
from typing import List, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── The one reused dependency: the Stage-2/3 error-pattern analysis ───────────
# Add src/ to sys.path so `cpt_decoder` resolves as a top-level package
# regardless of where this script is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_HERE, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ── Config helpers ────────────────────────────────────────────────────────────
def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


# ── Data: paths + cleaning (inlined from cpt_decoder.data.loader) ─────────────
DATA_DIR = os.path.join(_HERE, "src", "cpt_decoder", "data")
CORPUS_CSV = os.path.join(DATA_DIR, "sentphonemepairs_LRS2_original.csv")
HOMO_CSV = os.path.join(DATA_DIR, "sentences_with_homophones_37374.csv")
NON_HOMO_CSV = os.path.join(DATA_DIR, "sentences_without_homophones_10790.csv")

# Split sizes (sum to the full 48,164-row corpus)
TRAIN_N, VAL_N, TEST_N = 45839, 1082, 1243


def clean_phoneme_seq(seq: str) -> str:
    """Normalise an ARPAbet phoneme sequence: drop <SOS>/<EOS>, strip stress
    digits, replace the <space> word-boundary marker with a real space (it's
    not in the Llama vocab), collapse whitespace. Mirrors loader.clean_phoneme_seq."""
    seq = seq.strip()
    seq = re.sub(r"<SOS>|<EOS>", "", seq)
    seq = re.sub(r"[012]", "", seq)
    seq = seq.replace("<space>", " ")
    seq = re.sub(r"\s+", " ", seq).strip()
    return seq


def clean_sentence(text: str) -> str:
    return text.strip().upper()


def load_corpus() -> pd.DataFrame:
    """Headerless 2-column corpus (sentence, phoneme sequence), original order."""
    df = pd.read_csv(CORPUS_CSV, header=None, names=["Sentence", "Phoneme Transcription"])
    df = df.dropna()
    df["sentence"] = df["Sentence"].apply(clean_sentence)
    df["phonemes_raw"] = df["Phoneme Transcription"].astype(str)
    df["phonemes"] = df["Phoneme Transcription"].apply(clean_phoneme_seq)
    return df[["sentence", "phonemes", "phonemes_raw"]].reset_index(drop=True)


def _load_sentence_set(path: str) -> set:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return set(df["Sentence"].apply(clean_sentence))


def split_corpus(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if TRAIN_N + VAL_N + TEST_N != n:
        raise ValueError(
            f"corpus has {n} rows but {TRAIN_N}+{VAL_N}+{TEST_N}="
            f"{TRAIN_N + VAL_N + TEST_N} do not sum to n.")
    return (df.iloc[:TRAIN_N].copy(),
            df.iloc[TRAIN_N:TRAIN_N + VAL_N].copy(),
            df.iloc[TRAIN_N + VAL_N:].copy())


# ── Metrics (inlined from cpt_decoder.evaluation.metrics) ─────────────────────
import jiwer          # noqa: E402
import sacrebleu      # noqa: E402


def normalise(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _evaluate(refs: List[str], hyps: List[str], label: str) -> dict:
    if not refs:
        return {}
    rn = [normalise(r) for r in refs]
    hn = [normalise(h) for h in hyps]
    wer = jiwer.wer(rn, hn)
    cer = jiwer.cer(rn, hn)
    bleu = sacrebleu.corpus_bleu(hn, [rn]).score / 100.0
    em = sum(r == h for r, h in zip(rn, hn)) / len(rn)
    return {"label": label, "n_sentences": len(refs),
            "WER": wer, "CER": cer, "BLEU-4": bleu, "Exact_Match": em}


def stratified_evaluate(refs: List[str], hyps: List[str],
                        homo_mask: List[bool]) -> dict:
    hr = [r for r, m in zip(refs, homo_mask) if m]
    hh = [h for h, m in zip(hyps, homo_mask) if m]
    nr = [r for r, m in zip(refs, homo_mask) if not m]
    nh = [h for h, m in zip(hyps, homo_mask) if not m]
    return {"overall": _evaluate(refs, hyps, "Overall"),
            "homophone": _evaluate(hr, hh, "Homophone"),
            "non_homophone": _evaluate(nr, nh, "Non-Homophone")}


def print_results(results: dict, title: str) -> None:
    width = 62
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    print(f"  {'Subset':<22} {'N':>6}  {'WER':>7}  {'CER':>7}  {'BLEU-4':>7}  {'EM':>7}")
    print("-" * width)
    for key in ["overall", "homophone", "non_homophone"]:
        r = results.get(key, {})
        if not r:
            continue
        print(f"  {r['label']:<22} {r['n_sentences']:>6}  "
              f"{r['WER']*100:>6.2f}%  {r['CER']*100:>6.2f}%  "
              f"{r['BLEU-4']:>7.4f}  {r['Exact_Match']*100:>6.2f}%")
    print("=" * width)
    h, n = results.get("homophone"), results.get("non_homophone")
    if h and n:
        print(f"\n  >> WER gap  (homo - non-homo): {(h['WER']-n['WER'])*100:+.2f}%")
        print(f"  >> EM gap   (non-homo - homo): {(n['Exact_Match']-h['Exact_Match'])*100:+.2f}%")
    print()


def save_metrics_csv(results: dict, path: str, model_tag: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = []
    for r in results.values():
        if not r:
            continue
        rows.append({"timestamp": ts, "model": model_tag, "subset": r["label"],
                     "n": r["n_sentences"], "WER": round(r["WER"]*100, 4),
                     "CER": round(r["CER"]*100, 4), "BLEU4": round(r["BLEU-4"], 4),
                     "ExactMatch": round(r["Exact_Match"]*100, 4)})
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"  Metrics saved to: {path}")


# ── Model: plain base model, NO LoRA (4-bit on CUDA, fp/bf16 on CPU) ──────────
USE_4BIT = torch.cuda.is_available()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bnb_compute_dtype() -> torch.dtype:
    override = os.environ.get("ZS_BNB_COMPUTE_DTYPE")
    if override:
        return getattr(torch, override)
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:               # Ampere+ has bf16 hardware; Pascal/Turing don't
            return torch.bfloat16
    return torch.float16


def load_plain_model(model_name: str):
    """Load the raw base causal LM. No PEFT, no LoRA, no embedding resize."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   # standard zero-shot recipe
    tokenizer.padding_side = "left"                 # causal LM: left-pad for batched generate

    compute_dtype = _bnb_compute_dtype()
    quant_config = None
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=compute_dtype if USE_4BIT else torch.float32,
        device_map="auto" if USE_4BIT else None,
        low_cpu_mem_usage=True,
    )
    if not USE_4BIT:
        model = model.to(DEVICE)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_name}  (plain base, no LoRA | 4-bit: {USE_4BIT} | "
          f"compute dtype: {compute_dtype} | device: {DEVICE})")
    print(f"  Total parameters: {total:,}")
    return tokenizer, model


# ── Decode + stream ───────────────────────────────────────────────────────────
def _build_prompt(cleaned_phonemes: str) -> str:
    return f"Phonemes: {cleaned_phonemes}\nText:"


def _count_jsonl(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def generate_split(tokenizer, model, df: pd.DataFrame, split_name: str,
                   batch_size: int, max_new_tokens: int, out_dir: str) -> List[dict]:
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, f"zeroshot_{split_name}_{len(df)}.jsonl")
    json_path = os.path.join(out_dir, f"zeroshot_{split_name}_{len(df)}.json")

    done = _count_jsonl(jsonl_path)
    if done:
        print(f"  Resuming {split_name}: {done}/{len(df)} rows already written; skipping them.")

    # Load whatever was already streamed so metrics cover the full split.
    results: List[dict] = []
    if done:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            results = [json.loads(l) for l in f if l.strip()]

    todo = df.iloc[done:].reset_index(drop=True)
    n = len(todo)
    if n:
        print(f"  Generating on {n} {split_name} examples (batch_size={batch_size})...")
    t0 = time.time()
    for start in range(0, n, batch_size):
        batch = todo.iloc[start:start + batch_size]
        prompts = [_build_prompt(p) for p in batch["phonemes"].tolist()]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=96).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        prompt_len = enc["input_ids"].shape[1]
        bt = time.time()
        for j, (raw_phon, ref_text) in enumerate(zip(batch["phonemes_raw"].tolist(),
                                                      batch["sentence"].tolist())):
            decoded = tokenizer.decode(gen[j][prompt_len:], skip_special_tokens=True).strip()
            rec = {"index": done + start + j,
                   "input_phonemes": " ".join(str(raw_phon).split()),
                   "expected_text": ref_text, "model_output": decoded,
                   "time_seconds": (time.time() - bt) / max(1, len(batch))}
            results.append(rec)
            with open(jsonl_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        d = done + start + len(batch)
        el = time.time() - t0
        per = el / max(1, d - done)
        print(f"    {split_name}: {d}/{len(df)}  ({per:.2f}s/sample, "
              f"ETA {per*(len(df)-d)/60:.1f} min)", flush=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "total_samples": len(results)}, f,
                  ensure_ascii=False, indent=2)
    print(f"  Wrote {len(results)} samples to {os.path.basename(json_path)}")
    return results


def write_side_by_side(results: List[dict], homo_mask: List[bool],
                       split_name: str, out_dir: str) -> Tuple[str, int, int]:
    path = os.path.join(out_dir, f"zeroshot_{split_name}_{len(results)}_view.txt")
    n_ok = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Zero-Shot baseline side-by-side: {split_name} ({len(results)} samples)\n")
        f.write("# STATUS=OK iff normalised prediction == normalised target.\n")
        f.write("# INDEX | STATUS | HOMO | INPUT_PHONEMES (<=80c) | PREDICTED | TARGET\n")
        f.write("# " + "-" * 110 + "\n")
        for r, is_homo in zip(results, homo_mask):
            ok = normalise(r["expected_text"]) == normalise(r["model_output"])
            n_ok += ok
            ph = r.get("input_phonemes", "")
            ph = (ph[:77] + "...") if len(ph) > 80 else ph
            f.write(f"{r['index']:>5} | {'OK  ' if ok else 'WRONG'} | "
                    f"{'H' if is_homo else '-'}  | {ph} | "
                    f"{r['model_output']} | {r['expected_text']}\n")
    return path, n_ok, len(results) - n_ok


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    model_name = _env("ZS_MODEL", "meta-llama/Llama-3.2-3B")
    split_choice = _env("ZS_SPLIT", "all").lower()
    out_dir = _env("ZS_OUTPUT_DIR", os.path.join(_HERE, "zero-shot", "baseline"))
    batch_size = _env_int("ZS_BATCH_SIZE", 8 if torch.cuda.is_available() else 1)
    max_new_tokens = _env_int("ZS_MAX_NEW_TOK", 34)
    do_error_analysis = _env("ZS_ERROR_ANALYSIS", "1") == "1"

    print("=" * 70)
    print("  Zero-Shot Baseline (ISOLATED, plain base model)")
    print("=" * 70)
    print(f"  Model      : {model_name}")
    print(f"  Split      : {split_choice}")
    print(f"  Output dir : {out_dir}")
    print(f"  Batch size : {batch_size}   Max new tokens: {max_new_tokens}")
    print(f"  Error analysis: {'on' if do_error_analysis else 'off'}")

    print("\nLoading corpus...")
    full_df = load_corpus()
    print(f"  Corpus rows: {len(full_df):,}")
    train_df, val_df, test_df = split_corpus(full_df)
    print(f"  Split sizes: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    homo_set = _load_sentence_set(HOMO_CSV)
    print(f"  Homophone sentences: {len(homo_set):,}")

    tokenizer, model = load_plain_model(model_name)

    splits = []
    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        if split_choice in ("all", name):
            splits.append((name, d))
    if not splits:
        raise ValueError(f"ZS_SPLIT must be all|train|val|test, got {split_choice!r}")

    for split_name, split_df in splits:
        homo_mask = [s in homo_set for s in split_df["sentence"].tolist()]
        print("\n" + "-" * 70)
        print(f"  Split: {split_name}  ({len(split_df):,} rows, "
              f"{sum(homo_mask):,} homophone / {len(homo_mask)-sum(homo_mask):,} non-homophone)")
        print("-" * 70)

        results = generate_split(tokenizer, model, split_df, split_name,
                                 batch_size, max_new_tokens, out_dir)
        refs = [r["expected_text"] for r in results]
        hyps = [r["model_output"] for r in results]

        eval_results = stratified_evaluate(refs, hyps, homo_mask)
        print_results(eval_results, f"{model_name} zero-shot baseline -- {split_name}")
        save_metrics_csv(eval_results, os.path.join(out_dir, "metrics_summary.csv"),
                         f"{model_name} zero-shot baseline {split_name}")

        view_path, n_ok, n_wrong = write_side_by_side(results, homo_mask, split_name, out_dir)
        print(f"  Side-by-side view: {view_path}  ({n_ok} OK / {n_wrong} WRONG)")

        if do_error_analysis:
            from cpt_decoder.evaluation import error_analysis  # lazy: heavy deps
            err_report = error_analysis.error_category_report(
                refs, hyps, homo_mask=homo_mask, tokenizer=tokenizer, model=model, use_llm=False)
            error_analysis.print_error_report(
                err_report, title=f"{model_name} zero-shot baseline -- {split_name} (error patterns)")
            err_path = os.path.join(out_dir, f"zeroshot_{split_name}_{len(results)}_error_report.json")
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump(err_report, f, indent=2, ensure_ascii=False, default=str)
            print(f"  Error report JSON: {err_path}")

    print("\n" + "=" * 70)
    print(f"  Done. Outputs in: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
