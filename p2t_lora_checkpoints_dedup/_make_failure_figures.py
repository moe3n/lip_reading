"""Render fig09 — three-way comparison of failure-mode distributions:

  - v1 auto-classifier (n=76, rule cascade with late boundary check)
  - v2 auto-classifier (n=76, tightened boundary rule, moved earlier)
  - manual audit (n=26, human-verified subset)

These should sit in the docx Section 6 alongside the original
failure-mode bar chart (fig03), replacing or augmenting it.

Inputs:
  analysis/tables/bucket_counts.csv              (v1)
  analysis/tables/bucket_counts_v2.csv           (v2)
  analysis/tables/bucket_counts_audited.csv      (n=26 manual)

Output:
  analysis/figures/fig09_failure_modes_v1_v2_audit.png
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
TABLES = ROOT / "analysis" / "tables"
FIG = ROOT / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load(path):
    with path.open(encoding="utf-8", newline="") as f:
        return {r["bucket"]: int(r["n"]) for r in csv.DictReader(f)
                if r["bucket"] != "TOTAL"}


def load_audited(path):
    """Audited CSV has no TOTAL row, just bucket/auto_n/manual_n/etc."""
    with path.open(encoding="utf-8", newline="") as f:
        return {r["bucket"]: int(r["manual_n"]) for r in csv.DictReader(f)
                if r["bucket"] != "TOTAL"}


def main():
    v1 = load(TABLES / "bucket_counts.csv")
    v2 = load(TABLES / "bucket_counts_v2.csv")
    audit = load_audited(TABLES / "bucket_counts_audited.csv")

    buckets = sorted(set(v1) | set(v2) | set(audit), key=lambda b: -v2.get(b, 0))
    v1_n = sum(v1.values())   # 76
    v2_n = sum(v2.values())   # 76
    au_n = sum(audit.values())  # 26

    v1_pct = [100.0 * v1.get(b, 0) / v1_n for b in buckets]
    v2_pct = [100.0 * v2.get(b, 0) / v2_n for b in buckets]
    au_pct = [100.0 * audit.get(b, 0) / au_n for b in buckets]

    x = np.arange(len(buckets))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 6))

    bars1 = ax.bar(x - w, v1_pct, w, label=f"v1 auto (n={v1_n})",
                   color="#9ecae1", edgecolor="#225522")
    bars2 = ax.bar(x,     v2_pct, w, label=f"v2 auto (n={v2_n})",
                   color="#3182bd", edgecolor="#225522")
    bars3 = ax.bar(x + w, au_pct, w, label=f"manual audit (n={au_n})",
                   color="#fdae6b", edgecolor="#884400")

    def annotate(bars, vals):
        for rect, v in zip(bars, vals):
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.6,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    annotate(bars1, v1_pct)
    annotate(bars2, v2_pct)
    annotate(bars3, au_pct)

    # Make room for the percentages above tall bars
    ax.set_ylim(0, max(max(v1_pct), max(v2_pct), max(au_pct)) * 1.18)

    pretty = {
        "homophone_substitution": "homophone subst.",
        "non_word_spelling": "non-word spelling",
        "boundary_hallucination": "boundary hallucination",
        "semantic_substitution": "semantic subst.",
        "digit_word_rendering": "digit/word rendering",
        "truncation": "truncation",
        "oov_target_substitution": "OOV-target subst.",
        "suffix_hallucination": "suffix hallucination",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([pretty.get(b, b) for b in buckets],
                       rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("% of EM-False rows", fontsize=12)
    ax.set_title(
        "Figure 9 — Failure-mode distribution: rule v1, rule v2, and\n"
        "manual audit on the 26-row subset",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG / "fig09_failure_modes_v1_v2_audit.png", dpi=150)
    plt.close(fig)
    print(f"wrote fig09_failure_modes_v1_v2_audit.png")


if __name__ == "__main__":
    main()