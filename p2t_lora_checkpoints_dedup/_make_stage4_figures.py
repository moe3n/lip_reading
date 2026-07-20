"""Render Stage 4 figures:

  fig16 — BERTScore F1 distribution on EM-False (76 rows)
  fig17 — Substitution-category breakdown (Homophone vs Other), EM-False slice
  fig18 — Priority-ranked issues: ranked bar chart of the top-10 issues
          by some proxy score (BERTScore preservation = 1 - issue F1 hit
          rate). For a static figure we just rank by impact (% of rows
          touched) using the evidence_value strings.

Reads tables in analysis/tables/. Writes figures to analysis/figures/.
"""

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
TABLES = ROOT / "analysis" / "tables"
FIGS = ROOT / "analysis" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def _read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── fig16: BERTScore F1 distribution on EM-False ─────────────────────────────
def fig16_bertscore():
    rows = _read_csv(TABLES / "semantic_similarity.csv")
    f1 = [float(r["bertscore_f1"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.hist(f1, bins=15, color="#3182bd", edgecolor="white")
    ax.axvline(0.90, color="#31a354", linestyle="--", label="0.90 threshold")
    ax.axvline(0.70, color="#de2d26", linestyle="--", label="0.70 threshold")
    mean_f1 = sum(f1) / len(f1)
    ax.axvline(mean_f1, color="#000", linestyle=":", label=f"mean={mean_f1:.3f}")
    ax.set_xlabel("BERTScore F1 (deberta-base-mnli, EM-False slice)")
    ax.set_ylabel("# EM-False rows")
    ax.set_title(
        "Step 8 — BERTScore F1 distribution on EM-False rows (n=76)\n"
        "Even when WER is non-zero, the hypothesis is semantically close."
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS / "fig16_bertscore_distribution.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


# ── fig17: Substitution-category breakdown (Stage 4 Step 7, EM-False) ────────
def fig17_sub_categories():
    # Use Stage 4's grammar_breakdown (word-level subs on EM-False rows)
    rows = _read_csv(TABLES / "grammar_breakdown.csv")
    em_f = [r for r in rows if r["exact_match"] == "False"]

    n_homo = sum(int(r["homophone_subs"] or 0) for r in em_f)
    n_near = sum(int(r["near_homophone_subs"] or 0) for r in em_f)
    n_other = sum(int(r["other_subs"] or 0) for r in em_f)

    labels = ["Homophone", "Near-homophone", "Other"]
    counts = [n_homo, n_near, n_other]
    colors = ["#08519c", "#9ecae1", "#bdbdbd"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, counts, color=colors)
    for b, c in zip(bars, counts):
        if c:
            ax.text(b.get_x() + b.get_width() / 2, c + 0.5,
                    str(c), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# substitutions (EM-False slice, n=85)")
    ax.set_title(
        "Step 7 — Substitution-category breakdown (Option 3 grammar path)\n"
        "Closed-class detector sees 4 Homophone subs, 81 'Other'."
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS / "fig17_substitution_categories.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


# ── fig18: Priority-ranked issues (Stage 4 Step 10) ──────────────────────────
def fig18_priority_ranked():
    rows = _read_csv(TABLES / "priority_ranked.csv")
    rows.sort(key=lambda r: int(r["rank"]))
    labels = [
        (r["issue"][:42] + "...") if len(r["issue"]) > 42 else r["issue"]
        for r in rows
    ]

    # Use the evidence_value field's leading integer (e.g. "8 of 25 ...")
    # as a rough proxy for impact when present; fall back to 1.
    impact = []
    for r in rows:
        m = re.search(r"\b(\d+)\s+of\s+(\d+)", r["evidence_value"])
        if m:
            impact.append(int(m.group(1)))
        else:
            impact.append(1)

    # rank is small (1..10) — invert so higher rank = longer bar
    bar_lens = [len(rows) + 1 - int(r["rank"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = range(len(rows))
    ax.barh(list(y), bar_lens, color="#3182bd", alpha=0.85)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"P{r['rank']}: {lbl}" for r, lbl in zip(rows, labels)],
                       fontsize=8)
    ax.set_xticks([])
    ax.invert_yaxis()
    for i, (b, ev) in enumerate(zip(bar_lens, [r["evidence_value"] for r in rows])):
        ax.text(b + 0.05, i, ev[:60] + ("..." if len(ev) > 60 else ""),
                va="center", fontsize=7, color="#444")
    ax.set_title(
        "Step 10 — Priority-ranked issues (P1 = highest)\n"
        "Bar length is rank-order signal; annotation is the evidence_value."
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    out = FIGS / "fig18_priority_ranked.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig16_bertscore()
    fig17_sub_categories()
    fig18_priority_ranked()