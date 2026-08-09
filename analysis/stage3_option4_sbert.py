"""Stage 3 Option 4 (sentence-embedding cosine similarity) on any failing-rows
file. Same model and bands as the noise run, so results are comparable.

    python analysis/stage3_option4_sbert.py --failing <csv> --out <dir>

The CSV needs target and prediction columns; is_homophone is used if present.
"""

import argparse
import csv
import json
import statistics as st
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

MODEL = "all-MiniLM-L6-v2"

ap = argparse.ArgumentParser()
ap.add_argument("--failing", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

rows = list(csv.DictReader(open(args.failing, encoding="utf-8")))
refs = [r["target"] for r in rows]
hyps = [r["prediction"] for r in rows]
homo = [str(r.get("is_homophone", "")).strip().lower() in ("true", "1") for r in rows]

print(f"encoding {len(rows)} pairs with {MODEL}...")
model = SentenceTransformer(MODEL)
er = model.encode(refs, convert_to_tensor=True, normalize_embeddings=True)
eh = model.encode(hyps, convert_to_tensor=True, normalize_embeddings=True)
cos = [float(util.cos_sim(er[i], eh[i])[0][0]) for i in range(len(rows))]

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "stage3_option4_sbert.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["target", "prediction", "is_homophone", "cosine_similarity"])
    for r, h, hm, s in zip(refs, hyps, homo, cos):
        w.writerow([r, h, hm, round(s, 4)])

hf = [s for s, hm in zip(cos, homo) if hm]
nf = [s for s, hm in zip(cos, homo) if not hm]
res = {
    "model": MODEL, "method": "sentence-embedding cosine similarity",
    "n": len(cos), "mean": round(st.mean(cos), 4), "median": round(st.median(cos), 4),
    "ge_0.90": sum(s >= 0.90 for s in cos), "ge_0.70": sum(s >= 0.70 for s in cos),
    "lt_0.50": sum(s < 0.50 for s in cos),
    "homophone_mean": round(st.mean(hf), 4) if hf else None,
    "non_homophone_mean": round(st.mean(nf), 4) if nf else None,
}
json.dump(res, open(out_dir / "stage3_option4_sbert_summary.json", "w"), indent=2)
print(f"\nmean {res['mean']}  median {res['median']}")
print(f">=0.90: {res['ge_0.90']}/{res['n']}  >=0.70: {res['ge_0.70']}/{res['n']}  "
      f"<0.50: {res['lt_0.50']}/{res['n']}")
print(f"homophone mean {res['homophone_mean']}  non-homophone mean {res['non_homophone_mean']}")
