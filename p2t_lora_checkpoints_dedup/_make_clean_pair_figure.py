"""Render fig11 — substitution-pair distribution on the manually-cleaned
homophone_substitution subset.

Shows the histogram of substitution-pair counts: how many pairs fire
exactly 1, 2, 3, ... times across the 18 clean rows. The point is
that there's no systematic confusable pair — it's a long tail of
singletons.

Inputs:
  analysis/tables/homophone_clean_pairs.csv

Output:
  analysis/figures/fig11_clean_pair_distribution.png
"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
TABLES = ROOT / "analysis" / "tables"
FIG = ROOT / "analysis" / "figures"

with (TABLES / "homophone_clean_pairs.csv").open(encoding="utf-8", newline="") as f:
    pairs = list(csv.DictReader(f))

counts = [int(r["count"]) for r in pairs]
hist = Counter(counts)

xs = sorted(hist.keys())
ys = [hist[x] for x in xs]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar([str(x) for x in xs], ys,
              color="#3182bd", edgecolor="#225522")

for rect, n, x in zip(bars, ys, xs):
    label = f"{n}" + (" pair" if n == 1 else " pairs")
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.05,
            label, ha="center", va="bottom", fontsize=10)

ax.set_xlabel("substitution count (times a (tgt_phon, hyp_phon) pair fires)",
              fontsize=11)
ax.set_ylabel("number of distinct pairs", fontsize=11)
ax.set_title(
    f"Figure 11 — Phoneme-pair distribution on the manually-cleaned\n"
    f"homophone_substitution subset (n=18 rows, 13 distinct pairs)",
    fontsize=12,
)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(ys) * 1.2)

fig.tight_layout()
fig.savefig(FIG / "fig11_clean_pair_distribution.png", dpi=150)
plt.close(fig)
print("wrote fig11_clean_pair_distribution.png")