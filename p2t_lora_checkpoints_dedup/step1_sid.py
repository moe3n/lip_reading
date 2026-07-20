"""Step 1 — SID breakdown on dedup beam-5 predictions.

Reads p2t_lora_checkpoints_dedup/predictions_beam5.csv, runs jiwer
process_words / process_characters on every (target, prediction) pair,
writes per-pair alignment to sid_per_pair.csv, and prints an aggregate
table overall + split by homophone membership.

No GPU, no new dependencies (only jiwer, already in the venv).
"""

import csv
from pathlib import Path

import jiwer

ROOT = Path(__file__).parent
PREDICTIONS = ROOT / "predictions_beam5.csv"
OUT_PER_PAIR = ROOT / "sid_per_pair.csv"


def main():
    with PREDICTIONS.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert header == ["target", "prediction", "is_homophone"], header
    data = rows[1:]
    n = len(data)
    assert n == 949, f"expected 949 data rows, got {n}"

    # Per-pair aggregation
    per_pair_rows = []
    totals = {
        "word": {"S": 0, "I": 0, "D": 0, "H": 0, "N_ref": 0, "N_hyp": 0},
        "char": {"S": 0, "I": 0, "D": 0, "H": 0, "N_ref": 0, "N_hyp": 0},
    }
    buckets = {
        "overall": {"word": {"S": 0, "I": 0, "D": 0, "H": 0, "N_ref": 0, "N_hyp": 0},
                     "char": {"S": 0, "I": 0, "D": 0, "H": 0, "N_ref": 0, "N_hyp": 0}},
        "homophone": {"word": {"S": 0, "I": 0, "D": 0, "H": 0, "N_ref": 0, "N_hyp": 0},
                       "char": {"S": 0, "I": 0, "D": 0, "H": 0, "N_ref": 0, "N_hyp": 0}},
        "non_homophone": {"word": {"S": 0, "I": 0, "D": 0, "H": 0, "N_ref": 0, "N_hyp": 0},
                            "char": {"S": 0, "I": 0, "D": 0, "H": 0, "N_ref": 0, "N_hyp": 0}},
    }

    exact_matches = 0

    for tgt, hyp, is_homo in data:
        is_homo_bool = is_homo.strip().lower() == "true"
        bucket_key = "homophone" if is_homo_bool else "non_homophone"

        if tgt == hyp:
            exact_matches += 1

        # Word-level alignment
        w_align = jiwer.process_words(tgt, hyp)
        # jiwer versions differ: some return counts (ints), some return lists of chunks.
        def _count(x):
            return x if isinstance(x, int) else len(x)
        w_s, w_i, w_d, w_h = _count(w_align.substitutions), _count(w_align.insertions), _count(w_align.deletions), _count(w_align.hits)
        w_stats = {
            "S": w_s, "I": w_i, "D": w_d, "H": w_h,
            "N_ref": w_h + w_s + w_d,
            "N_hyp": w_h + w_s + w_i,
        }
        # Char-level alignment
        c_align = jiwer.process_characters(tgt, hyp)
        c_s, c_i, c_d, c_h = _count(c_align.substitutions), _count(c_align.insertions), _count(c_align.deletions), _count(c_align.hits)
        c_stats = {
            "S": c_s, "I": c_i, "D": c_d, "H": c_h,
            "N_ref": c_h + c_s + c_d,
            "N_hyp": c_h + c_s + c_i,
        }

        for k, v in w_stats.items():
            totals["word"][k] += v
            buckets["overall"]["word"][k] += v
            buckets[bucket_key]["word"][k] += v
        for k, v in c_stats.items():
            totals["char"][k] += v
            buckets["overall"]["char"][k] += v
            buckets[bucket_key]["char"][k] += v

        per_pair_rows.append({
            "target": tgt,
            "prediction": hyp,
            "is_homophone": is_homo,
            "word_S": w_stats["S"],
            "word_I": w_stats["I"],
            "word_D": w_stats["D"],
            "word_H": w_stats["H"],
            "char_S": c_stats["S"],
            "char_I": c_stats["I"],
            "char_D": c_stats["D"],
            "char_H": c_stats["H"],
            "wer": (w_stats["S"] + w_stats["I"] + w_stats["D"]) / max(1, w_stats["N_ref"]),
            "cer": (c_stats["S"] + c_stats["I"] + c_stats["D"]) / max(1, c_stats["N_ref"]),
        })

    # Write per-pair CSV
    with OUT_PER_PAIR.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_pair_rows[0].keys()))
        w.writeheader()
        w.writerows(per_pair_rows)

    def fmt_row(label, stats):
        n = stats["N_ref"]
        wer = (stats["S"] + stats["I"] + stats["D"]) / max(1, n)
        return (
            f"{label:<15} n_ref={stats['N_ref']:>5}  "
            f"S={stats['S']:>4} I={stats['I']:>3} D={stats['D']:>3} "
            f"H={stats['H']:>5}  rate={wer*100:6.3f}%"
        )

    def fmt_bucket(name, b):
        lines = [f"  -- {name} (word) --"]
        lines.append(fmt_row("  word", b["word"]))
        lines.append(f"  -- {name} (char) --")
        lines.append(fmt_row("  char", b["char"]))
        return "\n".join(lines)

    print(f"n_pairs = {n}")
    print(f"exact_match_count = {exact_matches} ({exact_matches/n*100:.2f}%)\n")

    print("==== WORD LEVEL ====")
    print(fmt_row("overall", totals["word"]))
    print(fmt_row("homophone", buckets["homophone"]["word"]))
    print(fmt_row("non_homophone", buckets["non_homophone"]["word"]))
    print("\n==== CHAR LEVEL ====")
    print(fmt_row("overall", totals["char"]))
    print(fmt_row("homophone", buckets["homophone"]["char"]))
    print(fmt_row("non_homophone", buckets["non_homophone"]["char"]))
    print(f"\nwrote per-pair alignment: {OUT_PER_PAIR}")


if __name__ == "__main__":
    main()
