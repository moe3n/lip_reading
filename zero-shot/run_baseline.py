"""
Zero-Shot Ablation Baseline — plain Llama-3.2-3B, phonemes -> text.

Purpose
-------
Lives outside src/ on purpose: this is an ablation study, not part of the
trained CPT decoder. No LoRA, no training, no import of model.py's PEFT
wrapper. It DOES reuse p2t_lora's data/eval utilities (loader.py,
g2p.py, metrics.py, evaluation/error_analysis.py) since those are generic
analysis tooling — plain refs/hyps in, numbers out — not decoder code.

Modes (run back-to-back on one model load, so results are comparable)
---------------------------------------------------------------------
    clean — loader.py's cleaned phonemes: no <SOS>/<EOS>/<space>/stress
            markers, e.g. "DH AH0 <space> K AE1 T" -> "DH A K AE T"
    raw   — loader.py's phonemes_raw, untouched: markers and stress digits
            intact, e.g. "<SOS> DH AH0 <space> K AE1 T <EOS>". Since those
            markers mean nothing to Llama's tokenizer on their own, "raw"
            uses a longer instruction that spells out the notation.

Metrics
-------
WER, CER, PER (phoneme error rate, via G2P round-trip), BLEU-4,
Exact Match — stratified Overall / Homophone / Non-Homophone, per mode.

Optional stages
---------------
Error analysis — every WER substitution classified as Homophone /
Near-homophone / Other (Stage 2), escalated through grammar-based
resolution (Stage 3 Option 3) where possible. Set ZS_ERROR_ANALYSIS=0 to
skip (the near-homophone lookup brute-scans the ~125k-word CMU dict,
~1s/substitution — expensive on the full 45,839-row train split).

Extended metrics (evaluation/extended_metrics.py, mirrors Mira Fleite's
prompting-baseline suite — see comparison/recompute_mira.py): SID
(substitution/insertion/deletion) breakdown, AER (place/manner/voicing),
and WPER (heuristic + panphon), stratified the same way as the core
metrics. Set ZS_EXTENDED_METRICS=0 to skip.

Env config (all optional)
-------------------------
    ZS_MODEL             default: meta-llama/Llama-3.2-3B
    ZS_SPLIT              "train" | "val" | "test"      default: "val"
    ZS_LIMIT               0 = whole split; N = first N rows  default: 0
    ZS_BATCH_SIZE          default: 8
    ZS_MODES               comma list of "clean","raw"   default: "clean,raw"
    ZS_ERROR_ANALYSIS      "1" | "0"                     default: "1"
    ZS_EXTENDED_METRICS    "1" | "0"                     default: "1"
    ZS_DEDUP_TRAIN         "1" | "0"                     default: "0"
    ZS_SPLIT_OFFSET        0-based row slice start        default: 0
    ZS_SPLIT_STRIDE        row slice step (1 = no slice)  default: 1

Resume
------
Predictions stream to preds_<split>_<n>_<mode>.jsonl, one mode per file.
Killing the process and re-running with the same split/mode picks up where
it left off instead of re-decoding.
"""

import os
import sys
import csv
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import jiwer

_HERE = os.path.dirname(os.path.abspath(__file__))          # zero-shot/
_SRC = os.path.join(os.path.dirname(_HERE), "src")
sys.path.insert(0, _SRC)

from p2t_lora.data import loader as data_loader           # noqa: E402
from p2t_lora.data import g2p                              # noqa: E402
from p2t_lora.evaluation.metrics import (                  # noqa: E402
    word_error_rate, character_error_rate, bleu4_score, exact_match,
)
from p2t_lora.evaluation.error_analysis import (           # noqa: E402
    error_category_report, print_error_report,
)
from p2t_lora.evaluation import extended_metrics as ext    # noqa: E402

# ── CONFIG ─────────────────────────────────────────────────────────────────
MODEL = os.environ.get("ZS_MODEL", "meta-llama/Llama-3.2-3B")
SPLIT = os.environ.get("ZS_SPLIT", "val")
LIMIT = int(os.environ.get("ZS_LIMIT", "0"))
BATCH_SIZE = int(os.environ.get("ZS_BATCH_SIZE", "8"))
MAX_NEW_TOKENS = 34
MODES = [m.strip() for m in os.environ.get("ZS_MODES", "clean,raw").split(",") if m.strip()]
RUN_ERROR_ANALYSIS = os.environ.get("ZS_ERROR_ANALYSIS", "1") == "1"
# SID/AER/WPER — cheap (jiwer + our own phoneme alignment, no model calls),
# on by default. Grammar (needs local Java) and semantic/BERTScore (real
# compute cost at scale) are deliberately NOT wired in here, same scope
# decision as comparison/recompute_mira.py — run them separately if wanted.
RUN_EXTENDED_METRICS = os.environ.get("ZS_EXTENDED_METRICS", "1") == "1"
# Train/eval sentence overlap: the LRS2 corpus repeats sentences (broadcast
# fillers, names), and a sequential slice keeps them on both sides of the
# boundary. Empirically 12.3% of val and 12.2% of test rows are verbatim
# duplicates of train rows (audit 2026-07-12). Setting ZS_DEDUP_TRAIN=1 drops
# any eval row whose sentence appears in train BEFORE decoding, so the
# reported metrics are on truly-novel data only. Off by default to keep the
# script's behaviour stable for the legacy 5k comparison.
DEDUP_TRAIN = os.environ.get("ZS_DEDUP_TRAIN", "0") == "1"

# Two-GPU parallel decode: split the chosen split (typically train, 45,839 rows)
# into N contiguous slices so N processes can each own one slice and write into
# disjoint output files. Each process sets ZS_SPLIT_OFFSET to its 0-based slot
# and ZS_SPLIT_STRIDE to N. Example for 2 GPUs: GPU0 gets OFFSET=0 STRIDE=2
# (even-indexed rows), GPU1 gets OFFSET=1 STRIDE=2 (odd-indexed rows). Stride=1
# is the legacy behaviour (no slicing). The offset is applied AFTER dedup, so
# dedup numbers stay comparable across processes.
SPLIT_OFFSET = int(os.environ.get("ZS_SPLIT_OFFSET", "0"))
SPLIT_STRIDE = int(os.environ.get("ZS_SPLIT_STRIDE", "1"))
if SPLIT_OFFSET < 0 or SPLIT_STRIDE < 1 or SPLIT_OFFSET >= SPLIT_STRIDE:
    raise ValueError(
        f"ZS_SPLIT_OFFSET={SPLIT_OFFSET} / ZS_SPLIT_STRIDE={SPLIT_STRIDE} "
        f"is invalid: need 0 <= OFFSET < STRIDE >= 1."
    )

TRAIN_N, VAL_N, TEST_N = 45839, 1082, 1243
OUT_DIR = os.path.join(_HERE, "baseline")

# Which loader.py column feeds the model in each mode.
PHONEME_COLUMN = {"clean": "phonemes", "raw": "phonemes_raw"}

INSTRUCTIONS = {
    "clean": (
        "You are given a sequence of ARPAbet phonemes representing one spoken "
        "English sentence. Convert the phonemes into the English sentence they "
        "spell out. Reply with only that sentence and nothing else."
    ),
    # The raw string still has <SOS>/<EOS>/<space>/stress-digit markup, which
    # means nothing to Llama's tokenizer unless we spell it out here.
    "raw": (
        "You are given a sequence of ARPAbet phonemes representing one spoken "
        "English sentence, using this notation: <SOS> marks the start and "
        "<EOS> marks the end of the sequence; <space> marks a boundary "
        "between words; a trailing digit on a vowel phoneme (0 = no stress, "
        "1 = primary stress, 2 = secondary stress) marks stress. Convert the "
        "phonemes into the English sentence they spell out. Reply with only "
        "that sentence and nothing else."
    ),
}


def build_prompt(mode: str, phonemes: str) -> str:
    return f"{INSTRUCTIONS[mode]}\n\nPhonemes: {phonemes}\nText:"


def extract_answer(text: str) -> str:
    # The base model answers on line 1, then rambles by re-emitting
    # "\n\nPhonemes: ...\nText: ..." forever. Keep only the first-line answer.
    text = text.split("\n", 1)[0]
    text = re.split(r"\bPhonemes\s*:", text)[0]
    return text.strip().strip('"').strip()


def phoneme_error_rate(refs, hyps) -> float:
    def to_ph(s):
        return " ".join(g2p.sentence_to_phoneme_list(s, stress=False))
    return jiwer.wer([to_ph(r) for r in refs], [to_ph(h) for h in hyps])


def stratify(refs, hyps, is_homo):
    """Split (refs, hyps) into the three subsets every metric in this script
    reports: Overall, Homophone, Non-Homophone."""
    homo = [(r, h) for r, h, m in zip(refs, hyps, is_homo) if m]
    non_homo = [(r, h) for r, h, m in zip(refs, hyps, is_homo) if not m]
    return {
        "Overall": (refs, hyps),
        "Homophone": tuple(zip(*homo)) if homo else ([], []),
        "Non-Homophone": tuple(zip(*non_homo)) if non_homo else ([], []),
    }


def score(refs, hyps, label):
    if not refs:
        return None
    return {
        "label": label, "n": len(refs),
        "WER": word_error_rate(refs, hyps) * 100,
        "CER": character_error_rate(refs, hyps) * 100,
        "PER": phoneme_error_rate(refs, hyps) * 100,
        "BLEU4": bleu4_score(refs, hyps),
        "EM": exact_match(refs, hyps) * 100,
    }


def extended(refs, hyps, label):
    if not refs:
        return None
    wper_panphon = None
    try:
        wper_panphon = ext.weighted_per(refs, hyps, method="panphon") * 100
    except RuntimeError as e:
        print(f"  WPER (panphon) skipped for {label}: {e}")
    return {
        "label": label,
        "sid": ext.sid_breakdown(refs, hyps),
        "aer": ext.allophonic_error_rate(refs, hyps),
        "wper_heuristic": ext.weighted_per(refs, hyps, method="heuristic") * 100,
        "wper_panphon": wper_panphon,
    }


def bnb_compute_dtype() -> torch.dtype:
    # bf16 needs compute capability >= 8.0 (Ampere+). GTX 1080 = Pascal (6.1).
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.bfloat16
    return torch.float16


def norm(t):
    t = (t or "").lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _patch_bnb_safe_to():
    """Monkeypatch PreTrainedModel.to so accelerate's `dispatch_model` can
    move a 4-bit bnb model. Idempotent — installs at most once.

    Why this is needed
    ------------------
    accelerate 1.14.0's `dispatch_model` resolves like this:

        use_multi_gpu = len(set(device_map.values())) > 1
        if use_multi_gpu and not check_cuda_p2p_ib_support():
            logger.warning(...)
        else:
            model.to(device)   # <-- crashes for 4-bit bnb

    With CUDA_VISIBLE_DEVICES pinning ONE GPU, `device_map="auto"` ends up
    with one unique device and accelerate calls `model.to(device)` on the
    whole 4-bit model, which raises:
        ValueError: .to is not supported for 4-bit or 8-bit bitsandbytes models

    With CUDA_VISIBLE_DEVICES empty (both GPUs visible) and device_map="auto",
    accelerate splits layers across both cards, hits the `use_multi_gpu=True`
    branch, and skips the .to() call entirely — that's what made the val-full
    smoke work. But splitting a 3B model across 2× GTX 1080 over PCIe is much
    slower than single-GPU (~0.3 rows/sec vs ~3+ rows/sec).

    transformers 4.44 unconditionally raises ValueError on `.to()` when
    `quantization_method == BITS_AND_BYTES`. accelerate's dispatch_model calls
    `.to(device)` to move ALL submodules (including non-Linear ones like
    `LlamaRotaryEmbedding` whose `inv_freq` buffer lives on CPU until `.to()`
    is called). Without that move, forward passes fail with
    "Expected all tensors to be on the same device, cpu and cuda:0".

    The fix: temporarily mask `quantization_method` while calling the original
    `.to()` so transformers lets the call through; bnb 0.43.2+ already
    supports moving 4-bit Linear layers itself, so the move itself is safe.
    """
    from transformers.modeling_utils import PreTrainedModel
    _orig = PreTrainedModel.to

    def _safe_to(self, *args, **kwargs):
        if (self.__class__.__name__ == "LlamaForCausalLM"
                or getattr(self, "is_loaded_in_4bit", False)
                or getattr(self, "is_loaded_in_8bit", False)):
            saved = self.__dict__.get("quantization_method", None)
            try:
                self.quantization_method = None
                return _orig(self, *args, **kwargs)
            finally:
                if saved is not None:
                    self.quantization_method = saved
        return _orig(self, *args, **kwargs)

    if not getattr(PreTrainedModel, "_bnb_safe_to_patched", False):
        PreTrainedModel.to = _safe_to
        PreTrainedModel._bnb_safe_to_patched = True


def load_model(tok):
    """Load the model in 4-bit bnb, pinned to the single visible GPU."""
    _patch_bnb_safe_to()
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bnb_compute_dtype(),
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=quant, device_map="auto",
    )
    model.eval()
    placed = next(model.parameters()).device
    print(f"  Loaded {MODEL} (4-bit, compute dtype {bnb_compute_dtype()}, "
          f"placed {placed})")
    return model


def required_input_len(tok, prompts) -> int:
    """Longest tokenized prompt actually present in `prompts` — the exact
    token budget needed for zero truncation, computed from the real data
    instead of a guessed constant. Truncating "Phonemes: ...\\nText:" away
    entirely is what caused raw mode to collapse to one fixed completion
    regardless of input (see zero-shot/analysis/results/ — pre-fix diagnosis)."""
    return max(len(tok(p, truncation=False)["input_ids"]) for p in prompts)


def run_mode(mode, tok, model_holder, split_df, is_homo, max_input_len):
    """Decode+score+error-analyse one preprocessing mode. model_holder is a
    single-element list acting as a lazy, shared mutable slot for the model
    so it's only loaded once, on whichever mode first needs it."""
    print("\n" + "=" * 70)
    print(f"  MODE: {mode}")
    print("=" * 70)

    col = PHONEME_COLUMN[mode]
    if SPLIT_STRIDE > 1:
        jsonl_path = os.path.join(
            OUT_DIR,
            f"preds_{SPLIT}_{len(split_df) * SPLIT_STRIDE}_{mode}_offset{SPLIT_OFFSET}-stride{SPLIT_STRIDE}.jsonl",
        )
    else:
        jsonl_path = os.path.join(OUT_DIR, f"preds_{SPLIT}_{len(split_df)}_{mode}.jsonl")

    already = 0
    if os.path.isfile(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            already = sum(1 for _ in f)
    work_df = split_df.iloc[already:].reset_index(drop=True)
    if already:
        print(f"  Resuming: {already}/{len(split_df)} rows already decoded, skipping them.")

    if len(work_df):
        if model_holder[0] is None:
            model_holder[0] = load_model(tok)
        model = model_holder[0]

        with open(jsonl_path, "a", encoding="utf-8", newline="\n") as out_f:
            for i in range(0, len(work_df), BATCH_SIZE):
                batch = work_df.iloc[i:i + BATCH_SIZE]
                prompts = [build_prompt(mode, p) for p in batch[col]]
                enc = tok(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_input_len).to(model.device)
                with torch.no_grad():
                    gen = model.generate(
                        **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                    )
                prompt_len = enc["input_ids"].shape[1]
                for j in range(len(batch)):
                    decoded = tok.decode(gen[j][prompt_len:], skip_special_tokens=True)
                    record = {
                        "index": already + i + j,
                        "phonemes": batch[col].iloc[j],
                        "target": batch["sentence"].iloc[j],
                        "prediction": extract_answer(decoded),
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                print(f"  {min(i + BATCH_SIZE, len(work_df)) + already}/{len(split_df)}", flush=True)
    else:
        print("  All rows already decoded, skipping generation.")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    assert len(records) == len(split_df), (
        f"{mode}: jsonl has {len(records)} rows, expected {len(split_df)} — "
        f"resume state doesn't match the requested split/limit."
    )

    refs = [r["target"] for r in records]
    hyps = [r["prediction"] for r in records]
    subsets = stratify(refs, hyps, is_homo)

    rows = [r for r in (score(*subsets[label], label) for label in subsets) if r]
    for r in rows:
        r["mode"] = mode

    print(f"\n{'Subset':<16}{'N':>6}{'WER':>9}{'CER':>9}{'PER':>9}{'BLEU4':>8}{'EM':>8}")
    for r in rows:
        print(f"{r['label']:<16}{r['n']:>6}{r['WER']:>8.2f}%{r['CER']:>8.2f}%"
              f"{r['PER']:>8.2f}%{r['BLEU4']:>8.4f}{r['EM']:>7.2f}%")

    with open(os.path.join(OUT_DIR, f"view_{SPLIT}_{len(split_df)}_{mode}.txt"), "w", encoding="utf-8") as f:
        f.write("STATUS | HOMO | PREDICTED | TARGET\n")
        for ref, hyp, m in zip(refs, hyps, is_homo):
            ok = "OK   " if norm(ref) == norm(hyp) else "WRONG"
            f.write(f"{ok} | {'H' if m else '-'} | {hyp} | {ref}\n")

    if RUN_ERROR_ANALYSIS:
        print(f"\n  Running error pattern analysis for '{mode}' (Stage 2/3 — this can take a while)...", flush=True)
        report = error_category_report(refs, hyps, homo_mask=is_homo)
        print_error_report(report, title=f"{MODEL} zero-shot [{mode}] -- {SPLIT}")
        with open(os.path.join(OUT_DIR, f"errors_{SPLIT}_{len(split_df)}_{mode}.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    if RUN_EXTENDED_METRICS:
        print(f"\n  Running extended metrics for '{mode}' (SID/AER/WPER)...", flush=True)
        ext_rows = [e for e in (extended(*subsets[label], label) for label in subsets) if e]
        for e in ext_rows:
            aer = e["aer"]
            wper_p = f"{e['wper_panphon']:.2f}%" if e["wper_panphon"] is not None else "skipped"
            print(f"  [{e['label']}] WPER heuristic={e['wper_heuristic']:.2f}%  panphon={wper_p}")
            print(f"      AER: place={aer['place_pct']:.1f}%  manner={aer['manner_pct']:.1f}%  "
                  f"voicing={aer['voicing_pct']:.1f}%  (of {aer['n_classified']} classified substitutions)")
        with open(os.path.join(OUT_DIR, f"extended_{SPLIT}_{len(split_df)}_{mode}.json"), "w", encoding="utf-8") as f:
            json.dump(ext_rows, f, indent=2, ensure_ascii=False, default=str)

    return rows


def main():
    print("=" * 70)
    print("  Zero-Shot Ablation Baseline (no LoRA, no training)")
    print("=" * 70)
    print(f"  Model  : {MODEL}")
    print(f"  Split  : {SPLIT}   Limit: {LIMIT or 'none (full split)'}")
    print(f"  Modes  : {', '.join(MODES)}")
    print(f"  Error analysis: {'ON' if RUN_ERROR_ANALYSIS else 'OFF'}")
    print(f"  Dedup vs train: {'ON (drop eval rows whose sentence is in train)' if DEDUP_TRAIN else 'OFF'}")
    if SPLIT_STRIDE > 1:
        print(f"  Slice  : offset={SPLIT_OFFSET} stride={SPLIT_STRIDE} (this process owns every {SPLIT_STRIDE}th row starting at index {SPLIT_OFFSET})")

    df = data_loader.load_original_phoneme_text_pairs()
    slices = {
        "train": df.iloc[:TRAIN_N],
        "val": df.iloc[TRAIN_N:TRAIN_N + VAL_N],
        "test": df.iloc[TRAIN_N + VAL_N:],
    }
    split_df = slices[SPLIT].reset_index(drop=True)
    if LIMIT:
        split_df = split_df.head(LIMIT)

    if DEDUP_TRAIN and SPLIT in ("val", "test"):
        train_sents = set(slices["train"]["sentence"])
        before = len(split_df)
        mask = ~split_df["sentence"].isin(train_sents)
        split_df = split_df[mask].reset_index(drop=True)
        print(f"  Dedup : dropped {before - len(split_df)}/{before} rows "
              f"(verbatim in train); {len(split_df)} kept.")

    # Apply ZS_SPLIT_OFFSET / ZS_SPLIT_STRIDE partitioning AFTER dedup so each
    # process sees the same dedup numbers and the union of all processes'
    # slices == dedup'd split. Two-GPU launch: GPU0 OFFSET=0 STRIDE=2,
    # GPU1 OFFSET=1 STRIDE=2. Each process gets a slice-tagged jsonl path so
    # they don't race on append.
    if SPLIT_STRIDE > 1:
        before_slice = len(split_df)
        split_df = split_df.iloc[SPLIT_OFFSET::SPLIT_STRIDE].reset_index(drop=True)
        print(f"  Slice : offset={SPLIT_OFFSET} stride={SPLIT_STRIDE} "
              f"({before_slice} -> {len(split_df)} rows for this process)")

    homo_set = set(data_loader.load_homophone_sentences()["sentence"])
    is_homo = [s in homo_set for s in split_df["sentence"]]
    print(f"  Rows   : {len(split_df)}  ({sum(is_homo)} homophone / {len(is_homo) - sum(is_homo)} non)")

    os.makedirs(OUT_DIR, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # Backstop: truncate from the left (eats into the instruction) so
    # "Phonemes: ...\nText:" at the tail is never the part that gets cut
    # away, on the off chance a row exceeds max_input_len below.
    tok.truncation_side = "left"

    all_prompts = [build_prompt(mode, p) for mode in MODES for p in split_df[PHONEME_COLUMN[mode]]]
    max_input_len = required_input_len(tok, all_prompts)
    print(f"  Max input length required (zero truncation): {max_input_len} tokens")

    model_holder = [None]  # loaded lazily by run_mode(), shared across modes
    all_rows = []
    for mode in MODES:
        all_rows.extend(run_mode(mode, tok, model_holder, split_df, is_homo, max_input_len))

    fieldnames = ["mode", "label", "n", "WER", "CER", "PER", "BLEU4", "EM"]
    # Per-split, per-size metrics CSV — does NOT clobber the legacy
    # `metrics.csv` (root name + 5k train/val/smoke only). Multiple runs
    # on different (split, n) now coexist as separate files.
    metrics_path = os.path.join(OUT_DIR, f"metrics_{SPLIT}_{len(split_df)}.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nMetrics written to {metrics_path}")

    if len(MODES) > 1:
        print("\n" + "=" * 70)
        print("  COMPARISON")
        print("=" * 70)
        print(f"{'Mode':<8}{'Subset':<16}{'N':>6}{'WER':>9}{'CER':>9}{'PER':>9}{'BLEU4':>8}{'EM':>8}")
        for r in all_rows:
            print(f"{r['mode']:<8}{r['label']:<16}{r['n']:>6}{r['WER']:>8.2f}%{r['CER']:>8.2f}%"
                  f"{r['PER']:>8.2f}%{r['BLEU4']:>8.4f}{r['EM']:>7.2f}%")

    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
