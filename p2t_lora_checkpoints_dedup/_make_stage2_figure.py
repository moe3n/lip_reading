"""Render fig10 — bucket leakage within v2's homophone_substitution
classification.

Shows how the 33 rows v2 tagged as 'homophone_substitution' break
down under manual reclassification. Only 18 are real single-word
phoneme swaps; 15 are leakage into boundary/suffix/semantic/truncation/
digit buckets.

Inputs:
  analysis/tables/homophone_manual_reclass.csv

Output:
  analysis/figures/fig10_homophone_bucket_leakage.png
"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
TABLES = ROOT / "analysis" / "tables"
FIG = ROOT / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

with (TABLES / "homophone_manual_reclass.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

true_counts = Counter(r["true_bucket"] for r in rows)
v2_counts = Counter(r["v2_bucket"] for r in rows)

buckets = sorted(set(true_counts) | set(v2_counts),
                 key=lambda b: -true_counts.get(b, 0))
v2_n = [v2_counts.get(b, 0) for b in buckets]
true_n = [true_counts.get(b, 0) for b in buckets]
total = len(rows)

pretty = {
    "homophone_substitution": "homophone subst. (clean)",
    "boundary_hallucination": "boundary hallucination",
    "suffix_hallucination":   "suffix hallucination",
    "semantic_substitution":  "semantic subst.",
    "truncation":             "truncation",
    "digit_word_rendering":   "digit/word rendering",
}
labels = [pretty.get(b, b) for b in buckets]

x = np.arange(len(buckets))
w = 0.4
fig, ax = plt.subplots(figsize=(10, 5.5))

bars_v2 = ax.bar(x - w/2, v2_n, w, label="v2 auto-bucket",
                 color="#3182bd", edgecolor="#225522")
bars_t = ax.bar(x + w/2, true_n, w, label="manual reclassification",
                color="#fdae6b", edgecolor="#884400")

for rect, n in zip(bars_v2, v2_n):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.15,
            f"{n}", ha="center", va="bottom", fontsize=9)
for rect, n in zip(bars_t, true_n):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.15,
            f"{n}", ha="center", va="bottom", fontsize=9)

# Mark "true" homophone vs the rest
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
ax.set_ylabel("count (of 33 v2-tagged homophone rows)", fontsize=11)
ax.set_title(
    f"Figure 10 — v2 auto-bucket vs manual reclassification within the\n"
    f"homophone_substitution bucket (n={total} EM-False rows)",
    fontsize=12,
)
ax.legend(loc="upper right", fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(max(v2_n), max(true_n)) * 1.18)

# Caption annotation
pct_clean = 100.0 * true_counts.get("homophone_substitution", 0) / total
ax.text(0.02, 0.92,
        f"Only {true_counts.get('homophone_substitution', 0)}/{total} "
        f"({pct_clean:.0f}%) are clean\nsingle-word phoneme swaps;\n"
        f"{total - true_counts.get('homophone_substitution', 0)}/{total} "
        f"are bucket leakage",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffcc",
                  edgecolor="#888888"))

fig.tight_layout()
fig.savefig(FIG / "fig10_homophone_bucket_leakage.png", dpi=150)
plt.close(fig)
print("wrote fig10_homophone_bucket_leakage.png")