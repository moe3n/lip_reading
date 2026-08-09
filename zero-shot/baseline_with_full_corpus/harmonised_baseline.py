"""
Harmonised Common-Parameters Zero-Shot Baseline (full corpus, raw mode).

Decoding parameters (frozen for this run)
-----------------------------------------
    do_sample          = False  (greedy)
    max_new_tokens     = 50
    num_beams          = 1
    repetition_penalty = 1.0  (none)
    temperature        = 0   (no-op under greedy)
    top_p              = 1   (no-op under greedy)
    seed               = 42

Data
----
    sentphonemepairs_LRS2_original.csv  (48,164 rows, headerless, original order)
    Phoneme column: "phonemes_raw"

Prompt
------
    Convert the following ARPAbet phoneme string to English text. Output
    only the English words, no extra text.
    Phonemes: {phonemes}
    English text:

Resume
------
Predictions stream to preds_<split>_<n>.jsonl, one row per line. Killing
and re-running with the same split/mode picks up where it left off.

Sharding (optional, 2 GPUs)
---------------------------
Set BWFC_OFFSET / BWFC_STRIDE to split the corpus across processes.
Example (2 GPUs, stride 2):
    GPU0:  CUDA_VISIBLE_DEVICES=0 BWFC_OFFSET=0 BWFC_STRIDE=2 ...
    GPU1:  CUDA_VISIBLE_DEVICES=1 BWFC_OFFSET=1 BWFC_STRIDE=2 ...
Run merge_shards.py after both shards finish to assemble metrics.
"""

import csv
import json
import os
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "..", "src")
sys.path.insert(0, _SRC)

from p2t_lora.data import loader as data_loader           # noqa: E402
from p2t_lora.data import g2p                              # noqa: E402
from p2t_lora.evaluation.metrics import (                  # noqa: E402
    word_error_rate, character_error_rate, bleu4_score, exact_match,
)
import jiwer                                                # noqa: E402


# ── Frozen harmonised parameters ──────────────────────────────────────────────
MODEL_NAME         = "meta-llama/Llama-3.2-3B"
DO_SAMPLE          = False
MAX_NEW_TOKENS     = 50
NUM_BEAMS          = 1
REPETITION_PENALTY = 1.0
TEMPERATURE        = 0.0
TOP_P              = 1.0
SEED               = 42
BATCH_SIZE         = int(os.environ.get("BWFC_BATCH_SIZE", "8"))
# Optional: shard the corpus across N processes (e.g. 2 GPUs).
BWFC_OFFSET        = int(os.environ.get("BWFC_OFFSET", "0"))
BWFC_STRIDE        = int(os.environ.get("BWFC_STRIDE", "1"))
PROMPT_TEMPLATE = (
    "Convert the following ARPAbet phoneme string to English text. "
    "Output only the English words, no extra text.\n"
    "Phonemes: {phonemes}\n"
    "English text:"
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bnb_compute_dtype() -> torch.dtype:
    # bf16 needs compute capability >= 8.0 (Ampere+). GTX 1080 = Pascal (6.1).
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.bfloat16
    return torch.float16


def _patch_bnb_safe_to() -> None:
    """Monkeypatch PreTrainedModel.to so accelerate's dispatch_model can
    move a 4-bit bnb model. Idempotent. See zero-shot/run_baseline.py for
    the full rationale (Pascal + single-GPU + device_map='auto' pitfall).
    """
    from transformers.modeling_utils import PreTrainedModel
    if getattr(PreTrainedModel, "_bnb_safe_to_patched", False):
        return
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

    PreTrainedModel.to = _safe_to
    PreTrainedModel._bnb_safe_to_patched = True


def load_model():
    _patch_bnb_safe_to()
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bnb_compute_dtype(),
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=quant, device_map="auto",
    )
    model.eval()
    placed = next(model.parameters()).device
    print(f"  Loaded {MODEL_NAME} (4-bit, compute dtype {bnb_compute_dtype()}, placed {placed})")
    return model


def extract_answer(text: str) -> str:
    # Base model may emit newlines after the answer; keep only the first line.
    text = text.split("\n", 1)[0]
    text = re.split(r"\bPhonemes\s*:", text)[0]
    return text.strip().strip('"').strip()


def phoneme_error_rate(refs, hyps) -> float:
    def to_ph(s):
        return " ".join(g2p.sentence_to_phoneme_list(s, stress=False))
    return jiwer.wer([to_ph(r) for r in refs], [to_ph(h) for h in hyps])


def norm(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def stratify(refs, hyps, is_homo):
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


def required_input_len(tok, prompts) -> int:
    return max(len(tok(p, truncation=False)["input_ids"]) for p in prompts)


def main():
    _seed_everything(SEED)

    print("=" * 70)
    print("  Harmonised Zero-Shot Baseline (full corpus, clean mode)")
    print("=" * 70)
    print(f"  Model           : {MODEL_NAME}")
    print(f"  Modes           : clean only")
    print(f"  do_sample       : {DO_SAMPLE}")
    print(f"  max_new_tokens  : {MAX_NEW_TOKENS}")
    print(f"  num_beams       : {NUM_BEAMS}")
    print(f"  repetition_pen. : {REPETITION_PENALTY}")
    print(f"  temperature     : {TEMPERATURE}")
    print(f"  top_p           : {TOP_P}")
    print(f"  seed            : {SEED}")
    print(f"  batch_size      : {BATCH_SIZE}")
    print(f"  shard           : offset={BWFC_OFFSET} stride={BWFC_STRIDE}")

    df = data_loader.load_original_phoneme_text_pairs()
    n_total = len(df)
    print(f"  Corpus          : {n_total:,} rows  "
          f"(sentphonemepairs_LRS2_original.csv, original order)")
    print(f"  Phoneme column  : phonemes_raw")

    out_dir = _HERE
    os.makedirs(out_dir, exist_ok=True)

    sharded = BWFC_STRIDE > 1 or BWFC_OFFSET != 0
    suffix = "" if not sharded else f"_offset{BWFC_OFFSET}-stride{BWFC_STRIDE}"
    jsonl_path = os.path.join(out_dir, f"preds_full_{n_total}{suffix}.jsonl")

    if sharded:
        # Per-shard path: each process owns row indices {offset, offset+stride, ...}.
        # Re-derive original indices so the merge step can sort & re-join cleanly.
        original_indices = list(range(BWFC_OFFSET, n_total, BWFC_STRIDE))
        work_df = df.iloc[original_indices].reset_index(drop=True)
        print(f"  Shard owns {len(work_df):,} rows of {n_total:,} "
              f"(offset={BWFC_OFFSET}, stride={BWFC_STRIDE}).")
    else:
        already = 0
        if os.path.isfile(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                already = sum(1 for _ in f)
        work_df = df.iloc[already:].reset_index(drop=True)
        if already:
            print(f"  Resuming: {already}/{n_total} rows already decoded, skipping them.")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    all_prompts = [PROMPT_TEMPLATE.format(phonemes=p) for p in df["phonemes_raw"]]
    max_input_len = required_input_len(tok, all_prompts)
    print(f"  Max input length required (zero truncation): {max_input_len} tokens")

    model = load_model()
    print(f"  Decoding {len(work_df):,} rows at batch_size={BATCH_SIZE}...")

    with open(jsonl_path, "a", encoding="utf-8", newline="\n") as out_f:
        for i in range(0, len(work_df), BATCH_SIZE):
            batch = work_df.iloc[i:i + BATCH_SIZE]
            prompts = [PROMPT_TEMPLATE.format(phonemes=p) for p in batch["phonemes_raw"]]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_input_len).to(model.device)
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "do_sample": DO_SAMPLE,
                    "num_beams": NUM_BEAMS,
                    "pad_token_id": tok.pad_token_id,
                    "eos_token_id": tok.eos_token_id,
                }
                if NUM_BEAMS > 1:
                    # num_beams > 1 implicitly forces sampling off; the
                    # temperature/top_p kwargs below only matter when
                    # do_sample=True, but include them anyway so a future
                    # switch to sampling doesn't change the call signature.
                    pass
                if REPETITION_PENALTY != 1.0:
                    gen_kwargs["repetition_penalty"] = REPETITION_PENALTY
                if DO_SAMPLE:
                    gen_kwargs["temperature"] = TEMPERATURE
                    gen_kwargs["top_p"] = TOP_P
                gen = model.generate(**enc, **gen_kwargs)
            prompt_len = enc["input_ids"].shape[1]
            for j in range(len(batch)):
                decoded = tok.decode(gen[j][prompt_len:], skip_special_tokens=True)
                if sharded:
                    rec_index = original_indices[i + j]
                else:
                    rec_index = already + i + j
                record = {
                    "index": rec_index,
                    "phonemes_raw": batch["phonemes_raw"].iloc[j],
                    "target": batch["sentence"].iloc[j],
                    "prediction": extract_answer(decoded),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            done = min(i + BATCH_SIZE, len(work_df))
            if sharded:
                print(f"  shard {done}/{len(work_df)} (orig rows up to "
                      f"{original_indices[done - 1]})", flush=True)
            else:
                print(f"  {already + done}/{n_total}", flush=True)

    if sharded:
        print(f"\nShard finished. Wrote {len(work_df):,} rows to {jsonl_path}")
        print("Run merge_shards.py after both shards complete to assemble metrics.")
        return

    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    assert len(records) == n_total, (
        f"jsonl has {len(records)} rows, expected {n_total} — "
        f"resume state doesn't match."
    )

    refs = [r["target"] for r in records]
    hyps = [r["prediction"] for r in records]

    homo_set = set(data_loader.load_homophone_sentences()["sentence"])
    is_homo = [s in homo_set for s in refs]

    subsets = stratify(refs, hyps, is_homo)
    rows = [r for r in (score(*subsets[label], label) for label in subsets) if r]

    print(f"\n{'Subset':<16}{'N':>8}{'WER':>10}{'CER':>10}{'PER':>10}{'BLEU4':>10}{'EM':>10}")
    for r in rows:
        print(f"{r['label']:<16}{r['n']:>8}{r['WER']:>9.2f}%{r['CER']:>9.2f}%"
              f"{r['PER']:>9.2f}%{r['BLEU4']:>10.4f}{r['EM']:>9.2f}%")

    metrics_path = os.path.join(out_dir, f"metrics_full_{n_total}.csv")
    fieldnames = ["label", "n", "WER", "CER", "PER", "BLEU4", "EM"]
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nMetrics written to {metrics_path}")

    view_path = os.path.join(out_dir, f"view_full_{n_total}.txt")
    with open(view_path, "w", encoding="utf-8") as f:
        f.write("STATUS | HOMO | PREDICTED | TARGET\n")
        for ref, hyp, m in zip(refs, hyps, is_homo):
            ok = "OK   " if norm(ref) == norm(hyp) else "WRONG"
            f.write(f"{ok} | {'H' if m else '-'} | {hyp} | {ref}\n")
    print(f"Side-by-side view : {view_path}")


if __name__ == "__main__":
    main()
