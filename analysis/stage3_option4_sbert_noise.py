"""Stage 3 Option 4, done exactly as the framework describes: sentence
embeddings and a cosine similarity between the target and the prediction.

Uses a sentence-transformer (all-MiniLM-L6-v2), which encodes each sentence to
one vector, then cosine similarity per failing row. This is the literal
"sentence embeddings" method, distinct from the token-level BERTScore used
earlier.

Reads p2t_lora_checkpoints_noise/analysis/failing_rows.csv, writes per-row
cosine to that folder, prints the distribution.
"""

import csv
import json
import statistics as st
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

ROOT = Path("p2t_lora_checkpoints_noise/analysis")
FAIL = ROOT / "failing_rows.csv"
MODEL = "all-MiniLM-L6-v2"

rows = list(csv.DictReader(open(FAIL, encoding="utf-8")))
refs = [r["target"] for r in rows]
hyps = [r["prediction"] for r in rows]
homo = [str(r["is_homophone"]).strip().lower() in ("true", "1") for r in rows]

print(f"encoding {len(rows)} pairs with {MODEL}...")
model = SentenceTransformer(MODEL)
er = model.encode(refs, convert_to_tensor=True, normalize_embeddings=True)
eh = model.encode(hyps, convert_to_tensor=True, normalize_embeddings=True)
cos = [float(util.cos_sim(er[i], eh[i])[0][0]) for i in range(len(rows))]

with open(ROOT / "stage3_option4_sbert.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["target", "prediction", "is_homophone", "cosine_similarity"])
    for r, h, hm, s in zip(refs, hyps, homo, cos):
        w.writerow([r, h, hm, round(s, 4)])

hf = [s for s, hm in zip(cos, homo) if hm]
nf = [s for s, hm in zip(cos, homo) if not hm]
out = {
    "model": MODEL, "method": "sentence-embedding cosine similarity",
    "n": len(cos), "mean": round(st.mean(cos), 4), "median": round(st.median(cos), 4),
    "ge_0.90": sum(s >= 0.90 for s in cos), "ge_0.70": sum(s >= 0.70 for s in cos),
    "lt_0.50": sum(s < 0.50 for s in cos),
    "homophone_mean": round(st.mean(hf), 4) if hf else None,
    "non_homophone_mean": round(st.mean(nf), 4) if nf else None,
}
json.dump(out, open(ROOT / "stage3_option4_sbert_summary.json", "w"), indent=2)

print(f"\nmean {out['mean']}  median {out['median']}")
print(f">=0.90: {out['ge_0.90']}/{out['n']}  >=0.70: {out['ge_0.70']}/{out['n']}  "
      f"<0.50: {out['lt_0.50']}/{out['n']}")
print(f"homophone mean {out['homophone_mean']}  non-homophone mean {out['non_homophone_mean']}")
