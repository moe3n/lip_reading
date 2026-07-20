"""Add an exact_match column to the beam-5 predictions and to the per-pair SID CSV.

Both outputs are derived views; the source predictions_beam5.csv is not modified.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).parent
PREDICTIONS = ROOT / "predictions_beam5.csv"
PER_PAIR = ROOT / "sid_per_pair.csv"
OUT_PREDICTIONS = ROOT / "predictions_beam5_with_match.csv"


def normalise(s: str) -> str:
    # Conservative: strip whitespace, uppercase. The headline EM uses jiwer's
    # default (lowercase + punctuation-stripped) normalisation, but a strict
    # equality test is the most transparent "exact yes/no" — using it as the
    # primary column, with jiwer-based EM as a sanity check.
    return s.strip().upper()


def main():
    # 1) Add exact_match to per-pair CSV (compute from target/prediction cells)
    with PER_PAIR.open(encoding="utf-8", newline="") as f:
        per_pair_rows = list(csv.DictReader(f))

    exact_counts = {"exact": 0, "diff": 0}
    for r in per_pair_rows:
        em = normalise(r["target"]) == normalise(r["prediction"])
        r["exact_match"] = "True" if em else "False"
        exact_counts["exact" if em else "diff"] += 1

    if per_pair_rows:
        fields = list(per_pair_rows[0].keys())
        if "exact_match" not in fields:
            fields.append("exact_match")
        with PER_PAIR.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(per_pair_rows)
    print(f"sid_per_pair.csv: updated, exact={exact_counts['exact']} diff={exact_counts['diff']} (n={len(per_pair_rows)})")

    # 2) Add exact_match to a copy of predictions_beam5.csv
    with PREDICTIONS.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert header == ["target", "prediction", "is_homophone"], header
    data = rows[1:]

    em_counts = {"exact": 0, "diff": 0}
    out_rows = [header + ["exact_match"]]
    for tgt, hyp, is_homo in data:
        em = normalise(tgt) == normalise(hyp)
        em_counts["exact" if em else "diff"] += 1
        out_rows.append([tgt, hyp, is_homo, "True" if em else "False"])

    with OUT_PREDICTIONS.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerows(out_rows)
    print(f"predictions_beam5_with_match.csv: written, exact={em_counts['exact']} diff={em_counts['diff']} (n={len(data)})")


if __name__ == "__main__":
    main()
