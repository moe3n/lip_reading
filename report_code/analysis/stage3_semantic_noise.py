"""Stage 3.3 semantic similarity for the noise model, using the same BERTScore
model the clean-model report used (microsoft/deberta-base-mnli, layer 10) so the
two are directly comparable.

Reads p2t_lora_checkpoints_noise/analysis/failing_rows.csv, writes per-row F1 to
that folder, prints the distribution.
"""

import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path("p2t_lora_checkpoints_noise/analysis")
FAIL = ROOT / "failing_rows.csv"

rows = list(csv.DictReader(open(FAIL, encoding="utf-8")))
refs = [r["target"] for r in rows]
hyps = [r["prediction"] for r in rows]
homo = [str(r["is_homophone"]).strip().lower() in ("true", "1") for r in rows]

print(f"scoring {len(rows)} failing rows with deberta-base-mnli (layer 10)...")
from bert_score import score as bert_score
_, _, F1 = bert_score(hyps, refs, lang="en",
                      model_type="microsoft/deberta-base-mnli", num_layers=10,
                      verbose=False)
f1 = [float(x) for x in F1]

with open(ROOT / "stage3_semantic.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["target", "prediction", "is_homophone", "bertscore_f1"])
    for r, h, hm, s in zip(refs, hyps, homo, f1):
        w.writerow([r, h, hm, round(s, 4)])

hf = [s for s, hm in zip(f1, homo) if hm]
nf = [s for s, hm in zip(f1, homo) if not hm]
out = {
    "n": len(f1),
    "mean": round(st.mean(f1), 4),
    "median": round(st.median(f1), 4),
    "ge_0.90": sum(s >= 0.90 for s in f1),
    "ge_0.70": sum(s >= 0.70 for s in f1),
    "lt_0.50": sum(s < 0.50 for s in f1),
    "homophone_mean": round(st.mean(hf), 4) if hf else None,
    "non_homophone_mean": round(st.mean(nf), 4) if nf else None,
}
json.dump(out, open(ROOT / "stage3_semantic_summary.json", "w"), indent=2)

print(f"\nmean {out['mean']}  median {out['median']}")
print(f">=0.90: {out['ge_0.90']}/{out['n']}  >=0.70: {out['ge_0.70']}/{out['n']}  "
      f"<0.50: {out['lt_0.50']}/{out['n']}")
print(f"homophone mean {out['homophone_mean']}  non-homophone mean {out['non_homophone_mean']}")
