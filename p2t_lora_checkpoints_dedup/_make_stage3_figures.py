"""Render Stage 3 figures: fig12 (WPER vs PER), fig13 (AER),
fig14 (per-phoneme drill-down), fig15 (error-type breakdown).

Reads tables in analysis/tables/. Writes figures to analysis/figures/.
"""

import csv
from collections import Counter
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


# ── fig12: WPER vs PER ────────────────────────────────────────────────────────
def fig12_wper():
    rows = _read_csv(TABLES / "wper_breakdown.csv")
    # Skip em_true (zero on both axes, not informative).
    rows = [r for r in rows if r["group"] != "em_true"]
    labels = [r["group"] for r in rows]
    per = [float(r["per"]) for r in rows]
    wper = [float(r["wper_heuristic"]) for r in rows]

    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars_per = ax.bar([i - width / 2 for i in x], per, width,
                       label="PER (plain)", color="#9ecae1")
    bars_wper = ax.bar([i + width / 2 for i in x], wper, width,
                        label="WPER (heuristic)", color="#3182bd")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Error rate (%)")
    ax.set_title("Plain PER vs heuristic WPER, by subset (dedup beam-5, n=949)")
    ax.legend(loc="upper left")
    # Annotate ratios
    ratios = [float(r["wper_per_ratio"]) for r in rows]
    for i, (b, ratio) in enumerate(zip(bars_wper, ratios)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                 f"r={ratio:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS / "fig12_wper_vs_per.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


# ── fig13: AER by feature ─────────────────────────────────────────────────────
def fig13_aer():
    rows = _read_csv(TABLES / "aer_breakdown.csv")
    labels = [r["group"].replace("em_false_", "") for r in rows]
    place = [float(r["place_pct"]) for r in rows]
    manner = [float(r["manner_pct"]) for r in rows]
    voicing = [float(r["voicing_pct"]) for r in rows]
    n_subs = [int(r["n_substitutions"]) for r in rows]

    x = range(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar([i - width for i in x], place, width, label="place",
            color="#756bb1", alpha=0.9)
    ax.bar([i for i in x], manner, width, label="manner",
            color="#3182bd", alpha=0.9)
    ax.bar([i + width for i in x], voicing, width, label="voicing",
            color="#31a354", alpha=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{l}\n(n_subs={n})" for l, n in zip(labels, n_subs)])
    ax.set_ylabel("% of substitutions where feature differs")
    ax.set_title("Allophonic Error Rate (AER) by feature, on EM-False rows")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS / "fig13_aer_by_feature.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


# ── fig14: per-phoneme drill-down ─────────────────────────────────────────────
def fig14_per_phoneme():
    rows = _read_csv(TABLES / "per_phoneme_drilldown.csv")
    rows = [r for r in rows if r["low_confidence"] == "no"]
    rows.sort(key=lambda r: int(r["n_substitutions"]), reverse=True)
    labels = [r["ref_phon"] for r in rows]
    counts = [int(r["n_substitutions"]) for r in rows]
    rates = [float(r["within_phoneme_error_rate"]) for r in rows]
    top = [r["top_hyp_phon"] for r in rows]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))
    # Left: absolute substitution counts (high-N only).
    ax_l.barh(labels[::-1], counts[::-1], color="#3182bd")
    ax_l.set_xlabel("# substitutions (EM-False)")
    ax_l.set_title("Per-phoneme substitution count (low-N filtered)")
    for i, (n, hyp) in enumerate(zip(counts[::-1], top[::-1])):
        ax_l.text(n + 0.05, i, f" -> {hyp}", va="center", fontsize=8,
                   color="#555")
    ax_l.grid(axis="x", alpha=0.3)

    # Right: within-phoneme error rate (substitutions / ref_occurrences).
    ax_r.barh(labels[::-1], rates[::-1], color="#756bb1")
    ax_r.set_xlabel("Within-phoneme error rate (%)")
    ax_r.set_title("Per-phoneme error rate (substitutions / ref occurrences)")
    for i, (rate, ref_n) in enumerate(zip(rates[::-1],
                                            [int(r["ref_occurrences"]) for r in rows][::-1])):
        ax_r.text(rate + 0.3, i, f" ref_n={ref_n}", va="center", fontsize=8,
                   color="#555")
    ax_r.grid(axis="x", alpha=0.3)

    fig.suptitle("Step 6 — Per-phoneme drill-down on EM-False (n=76)", y=1.02)
    fig.tight_layout()
    out = FIGS / "fig14_per_phoneme_drilldown.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ── fig15: error-type breakdown ───────────────────────────────────────────────
def fig15_error_type():
    rows = _read_csv(TABLES / "error_type_breakdown.csv")
    labels = [r["label"] for r in rows]
    pct_all = [float(r["pct_all"]) for r in rows]
    pct_emf = [float(r["pct_em_false"]) for r in rows]

    x = range(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar([i - width / 2 for i in x], pct_all, width,
            label="overall (n=949)", color="#9ecae1")
    ax.bar([i + width / 2 for i in x], pct_emf, width,
            label="EM-False (n=76)", color="#08519c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("% of rows")
    ax.set_title("Error-type breakdown (fast variant: exact homophone only)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS / "fig15_error_type_breakdown.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig12_wper()
    fig13_aer()
    fig14_per_phoneme()
    fig15_error_type()