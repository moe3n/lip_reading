"""
Merge per-shard jsonl outputs from harmonised_baseline.py into the final
preds_full_<N>.jsonl + metrics_full_<N>.csv + view_full_<N>.txt.

Run after all shards (e.g. 2 GPUs, offset 0/1 stride 2) finish.

Usage:
    python merge_shards.py               # picks up all shards in this folder
    python merge_shards.py --force       # overwrite merged jsonl if it exists
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "..", "src")
sys.path.insert(0, _SRC)

from p2t_lora.data import loader as data_loader           # noqa: E402
from p2t_lora.data import g2p                              # noqa: E402
from p2t_lora.evaluation.metrics import (                  # noqa: E402
    word_error_rate, character_error_rate, bleu4_score, exact_match,
)
import jiwer                                                # noqa: E402

SHARD_RE = re.compile(r"preds_full_(\d+)_offset(\d+)-stride(\d+)\.jsonl$")


def extract_answer(text: str) -> str:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite merged jsonl even if it exists")
    args = ap.parse_args()

    # Find all shard files in this folder.
    candidates = []
    for path in glob.glob(os.path.join(_HERE, "preds_full_*_offset*-stride*.jsonl")):
        m = SHARD_RE.search(os.path.basename(path))
        if m:
            candidates.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), path))

    if not candidates:
        print("No shard jsonl files found. Expected names like "
              "preds_full_48164_offset0-stride2.jsonl")
        sys.exit(1)

    n_total = candidates[0][0]
    for n, _, _, _ in candidates:
        if n != n_total:
            print(f"Shards disagree on total N: {n} vs {n_total}")
            sys.exit(1)

    candidates.sort(key=lambda x: (x[2], x[1]))  # stride, then offset
    print(f"Found {len(candidates)} shard(s) for N={n_total}:")
    for n, off, st, p in candidates:
        print(f"  offset={off} stride={st}  {os.path.basename(p)}")

    expected_rows = sum(1 for _, _, _, _ in candidates) * 0
    expected_rows = 0
    for _, off, st, p in candidates:
        with open(p, "r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        # Rows of the original corpus this shard owns.
        n_rows = (n_total - off + st - 1) // st
        if n_lines != n_rows:
            print(f"  WARNING: {os.path.basename(p)} has {n_lines} rows, "
                  f"expected {n_rows} ({n_total} rows, offset={off}, stride={st})")
        expected_rows += n_lines
        print(f"    -> {n_lines} rows decoded")

    if expected_rows != n_total:
        print(f"\nTotal rows across shards = {expected_rows}, "
              f"but N = {n_total}. Missing shards?")
        sys.exit(1)

    merged_path = os.path.join(_HERE, f"preds_full_{n_total}.jsonl")
    if os.path.exists(merged_path) and not args.force:
        print(f"\n{merged_path} already exists. Re-run with --force to overwrite.")
        sys.exit(1)

    records = []
    for _, _, _, p in candidates:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["index"])

    if len(records) != n_total:
        print(f"Merged {len(records)} rows, expected {n_total}.")
        sys.exit(1)

    with open(merged_path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nMerged jsonl written to {merged_path}")

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

    metrics_path = os.path.join(_HERE, f"metrics_full_{n_total}.csv")
    fieldnames = ["label", "n", "WER", "CER", "PER", "BLEU4", "EM"]
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nMetrics written to {metrics_path}")

    view_path = os.path.join(_HERE, f"view_full_{n_total}.txt")
    with open(view_path, "w", encoding="utf-8") as f:
        f.write("STATUS | HOMO | PREDICTED | TARGET\n")
        for ref, hyp, m in zip(refs, hyps, is_homo):
            ok = "OK   " if norm(ref) == norm(hyp) else "WRONG"
            f.write(f"{ok} | {'H' if m else '-'} | {hyp} | {ref}\n")
    print(f"Side-by-side view : {view_path}")


if __name__ == "__main__":
    main()
