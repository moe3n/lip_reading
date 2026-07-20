"""Generate matplotlib figures for analysis/findings.docx.

All numbers are inlined from the docx builder so the figures stay
in sync with the prose when re-rendered.

Outputs:
  analysis/figures/fig01_beam5_vs_greedy.png
  analysis/figures/fig02_sid_stacked.png
  analysis/figures/fig03_failure_modes.png
  analysis/figures/fig04_real_word_breakdown.png
  analysis/figures/fig05_oov_target_vs_pred.png
  analysis/figures/fig06_per_vs_cer.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
FIG = ROOT / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Inlined so figures match prose ---------------------------------------------

HEADLINE = {
    "n": 949,
    "wer_greedy": 3.53,
    "wer_beam5": 2.09,
    "cer_beam5": 0.98,
    "bleu_beam5": 0.9673,
    "em_greedy": 86.51,
    "em_beam5": 91.99,
}

SID_ROWS = [
    # name, n_ref, S, I, D, H, rate
    ("Word · overall",       5214, 87, 13, 11, 5116, 2.129),
    ("Word · homophone",     3995, 57,  8,  9, 3929, 1.852),
    ("Word · non-homophone", 1219, 30,  5,  2, 1187, 3.035),
    ("Char · overall",      26345,116, 67, 78,26151, 0.991),
    ("Char · homophone",    19751, 74, 35, 60,19617, 0.856),
    ("Char · non-homophone", 6594, 42, 32, 18, 6534, 1.395),
]

BUCKETS = [
    ("homophone_substitution",   34),
    ("non_word_spelling",        17),
    ("semantic_substitution",     8),
    ("boundary_hallucination",    5),
    ("digit_word_rendering",      5),
    ("oov_target_substitution",   3),
    ("truncation",                3),
    ("suffix_hallucination",      1),
]

OOV_TARGET = 34
OOV_PRED = 56


# Per-row CER + PER for Figure 6 ---------------------------------------------
# CER is already in sid_per_pair.csv; PER is computed from the
# ARPAbet token sequences already stored on predictions_beam5_with_match.csv
# via jiwer's word-aligned edit-distance on phoneme tokens.

import csv

SID_CSV = ROOT / "sid_per_pair.csv"
PHON_CSV = ROOT / "predictions_beam5_with_match.csv"


def _phoneme_tokens(phon_str):
    if not phon_str:
        return []
    out = []
    for tok in phon_str.split():
        if tok.startswith("{") and tok.endswith("}"):
            out.append(tok)
        else:
            out.extend(tok)
    return [t for t in out if t and t != "|"]


def _edit_alignment(ref_toks, hyp_toks):
    """jiwer-style edit alignment on two lists. Returns (S, I, D, H)."""
    n, m = len(ref_toks), len(hyp_toks)
    # dp[i][j] = (s, i, d) cost tuple
    dp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0)
    for j in range(1, m + 1):
        dp[0][j] = (0, j, 0)
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            r, h = ref_toks[i - 1], hyp_toks[j - 1]
            if r == h:
                s, ins, d = dp[i - 1][j - 1]
                dp[i][j] = (s, ins, d)
            else:
                s_sub, ins_sub, d_sub = dp[i - 1][j - 1]
                s_del, ins_del, d_del = dp[i - 1][j]
                s_ins, ins_ins, d_ins = dp[i][j - 1]
                sub = (s_sub + 1, ins_sub, d_sub)
                delete = (s_del + 1, ins_del, d_del)
                insert = (s_ins, ins_ins + 1, d_ins)
                # Prefer substitution over insert+delete at equal cost
                best = sub
                if (insert[0] + insert[1] + insert[2]) < (best[0] + best[1] + best[2]):
                    best = insert
                if (delete[0] + delete[1] + delete[2]) < (best[0] + best[1] + best[2]):
                    best = delete
                dp[i][j] = best
    return dp[n][m]


def load_per_rows():
    """Returns list of dicts {wer, cer, per, exact_match} for the 949 rows."""
    import math

    sid_by_idx = {}
    with SID_CSV.open(encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            sid_by_idx[i] = row

    rows = []
    with PHON_CSV.open(encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            sid = sid_by_idx.get(i, {})
            wer = float(sid.get("wer") or 0.0)
            cer = float(sid.get("cer") or 0.0)
            em = sid.get("exact_match", "False") in ("True", "true", "1")
            ref = _phoneme_tokens(row.get("target_phonemes", ""))
            hyp = _phoneme_tokens(row.get("prediction_phonemes", ""))
            s, ins, d = _edit_alignment(ref, hyp)
            n_ref = len(ref)
            per = ((s + ins + d) / n_ref * 100.0) if n_ref > 0 else 0.0
            rows.append({"wer": wer, "cer": cer, "per": per,
                         "exact_match": em, "n_ref_phonemes": n_ref})
    rows = [r for r in rows if r["n_ref_phonemes"] >= 3]
    # overall mean PER
    overall_per = sum(r["per"] for r in rows) / len(rows)
    return rows, overall_per


PER_ROWS, OVERALL_PER = load_per_rows()

# Common style --------------------------------------------------------------

PALETTE = {
    "S":   "#4e79a7",  # blue
    "I":   "#f28e2b",  # orange
    "D":   "#e15759",  # red
    "H":   "#bab0ab",  # grey (hits)
    "Greedy": "#b0b0b0",
    "Beam-5": "#4e79a7",
    "EM_target": "#bab0ab",
    "EM_ours":   "#4e79a7",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def add_value_labels(ax, rects, fmt="{:.2f}", offset=0.4):
    for r in rects:
        h = r.get_height()
        ax.annotate(fmt.format(h),
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)


# Figure 1 — decoding strategy comparison -----------------------------------

def fig01_beam5_vs_greedy():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.2, 3.4))

    # WER (lower better)
    x = ["Greedy", "Beam-5"]
    wer = [HEADLINE["wer_greedy"], HEADLINE["wer_beam5"]]
    bars = axL.bar(x, wer, color=[PALETTE["Greedy"], PALETTE["Beam-5"]])
    add_value_labels(axL, bars, fmt="{:.2f}%")
    axL.set_ylabel("WER (%) ↓")
    axL.set_title("Word Error Rate")
    axL.set_ylim(0, max(wer) * 1.25)

    # EM (higher better)
    em = [HEADLINE["em_greedy"], HEADLINE["em_beam5"]]
    bars = axR.bar(x, em, color=[PALETTE["Greedy"], PALETTE["Beam-5"]])
    add_value_labels(axR, bars, fmt="{:.2f}%")
    axR.set_ylabel("Exact Match (%) ↑")
    axR.set_title("Exact Match")
    axR.set_ylim(0, 100)

    fig.suptitle(f"Beam-5 vs Greedy on the dedup val set (n={HEADLINE['n']})",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig01_beam5_vs_greedy.png")
    plt.close(fig)
    print("wrote fig01_beam5_vs_greedy.png")


# Figure 2 — SID stacked bars ------------------------------------------------

def fig02_sid_stacked():
    fig, ax = plt.subplots(figsize=(8.0, 4.0))

    names = [r[0] for r in SID_ROWS]
    S = [r[2] for r in SID_ROWS]
    I = [r[3] for r in SID_ROWS]
    D = [r[4] for r in SID_ROWS]
    H = [r[5] for r in SID_ROWS]

    import numpy as np
    x = np.arange(len(names))
    w = 0.55
    ax.bar(x, S, w, label="Substitution", color=PALETTE["S"])
    ax.bar(x, I, w, bottom=S, label="Insertion", color=PALETTE["I"])
    ax.bar(x, D, w, bottom=[s + i for s, i in zip(S, I)],
           label="Deletion", color=PALETTE["D"])
    # Hits stacked above on the same axis for visual sense of share
    ax.bar(x, H, w, bottom=[s + i + d for s, i, d in zip(S, I, D)],
           label="Hits", color=PALETTE["H"], alpha=0.55)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Token count")
    ax.set_title("Substitution / Insertion / Deletion / Hits per bucket")
    ax.legend(loc="upper right", ncols=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig02_sid_stacked.png")
    plt.close(fig)
    print("wrote fig02_sid_stacked.png")


# Figure 3 — failure-mode distribution --------------------------------------

def fig03_failure_modes():
    fig, ax = plt.subplots(figsize=(7.6, 4.0))

    names = [b[0] for b in BUCKETS]
    counts = [b[1] for b in BUCKETS]

    # Colour by family: real-word (homophone + semantic), then others
    family = []
    for n, _ in BUCKETS:
        if n in ("homophone_substitution", "semantic_substitution"):
            family.append("#4e79a7")
        elif n == "non_word_spelling":
            family.append("#f28e2b")
        elif n == "boundary_hallucination":
            family.append("#59a14f")
        elif n == "digit_word_rendering":
            family.append("#edc948")
        else:
            family.append("#bab0ab")

    y = list(range(len(names)))
    bars = ax.barh(y, counts, color=family)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Rows (EM-False = 76)")
    ax.set_title("Failure-mode distribution (EM-False rows)")
    ax.set_xlim(0, max(counts) * 1.18)
    for r, c in zip(bars, counts):
        pct = c / 76 * 100
        ax.annotate(f"{c} ({pct:.1f}%)",
                    xy=(r.get_width(), r.get_y() + r.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9)

    # Family legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#4e79a7", label="Real-word substitution"),
        Patch(facecolor="#f28e2b", label="Non-word / invented spelling"),
        Patch(facecolor="#59a14f", label="Boundary (split/merge)"),
        Patch(facecolor="#edc948", label="Digit ↔ word"),
        Patch(facecolor="#bab0ab", label="Truncation / OOV / suffix"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig03_failure_modes.png")
    plt.close(fig)
    print("wrote fig03_failure_modes.png")


# Figure 4 — real_word vs leave-alone ---------------------------------------

def fig04_real_word_breakdown():
    # 76 EM-False rows broken into three top-level buckets:
    #   real_word_substitution = homophone_substitution + semantic_substitution = 42
    #   non_word_spelling = 17
    #   structural = boundary + digit + oov_target + truncation + suffix = 5+5+3+3+1 = 17
    rw = 42
    nw = 17
    st = 17  # 76 - 42 - 17

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cats = ["Real-word\nsubstitution", "Non-word\nspelling", "Structural\n(spacing / digit / OOV / truncation / suffix)"]
    counts = [rw, nw, st]
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]
    bars = ax.bar(cats, counts, color=colors)
    for r, c in zip(bars, counts):
        pct = c / 76 * 100
        ax.annotate(f"{c}\n({pct:.1f}%)",
                    xy=(r.get_x() + r.get_width() / 2, r.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Rows (of 76 EM-False)")
    ax.set_ylim(0, max(counts) * 1.25)
    ax.set_title("Three-bucket collapse of EM-False rows")
    fig.tight_layout()
    fig.savefig(FIG / "fig04_real_word_breakdown.png")
    plt.close(fig)
    print("wrote fig04_real_word_breakdown.png")


# Figure 5 — OOV target vs prediction ---------------------------------------

def fig05_oov_target_vs_pred():
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    cats = ["Target contains\nat least one OOV word", "Prediction contains\nat least one OOV word"]
    counts = [OOV_TARGET, OOV_PRED]
    colors = [PALETTE["EM_target"], PALETTE["EM_ours"]]
    bars = ax.bar(cats, counts, color=colors)
    for r, c, n in zip(bars, counts, [949, 949]):
        pct = c / n * 100
        ax.annotate(f"{c} / {n}\n({pct:.1f}%)",
                    xy=(r.get_x() + r.get_width() / 2, r.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Rows")
    ax.set_ylim(0, max(counts) * 1.25)
    ax.set_title("Phoneme OOV: target vs prediction")
    fig.tight_layout()
    fig.savefig(FIG / "fig05_oov_target_vs_pred.png")
    plt.close(fig)
    print("wrote fig05_oov_target_vs_pred.png")


# Figure 6 — PER vs CER scatter ---------------------------------------------

def fig06_per_vs_cer():
    if not PER_ROWS:
        print("skipped fig06_per_vs_cer.png (no PER rows)")
        return

    em_pts = [(r["per"], r["cer"]) for r in PER_ROWS if r["exact_match"]]
    no_pts = [(r["per"], r["cer"]) for r in PER_ROWS if not r["exact_match"]]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    if em_pts:
        ax.scatter([p[0] for p in em_pts], [p[1] for p in em_pts],
                   s=14, alpha=0.55, color="#2ca02c", label="Exact match (873)")
    if no_pts:
        ax.scatter([p[0] for p in no_pts], [p[1] for p in no_pts],
                   s=26, alpha=0.85, color="#e15759",
                   edgecolor="#7f0000", linewidth=0.4,
                   label="EM-False (76)")

    # Identity reference
    lim = max(
        max(p[0] for p in em_pts + no_pts) if (em_pts or no_pts) else 0,
        max(p[1] for p in em_pts + no_pts) if (em_pts or no_pts) else 0,
    ) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color="#999999", lw=1, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    ax.set_xlabel("PER (%) — phoneme-level edit distance")
    ax.set_ylabel("CER (%) — character-level edit distance")
    ax.set_title(f"PER vs CER per row (n={len(PER_ROWS)}, mean PER={OVERALL_PER:.2f}%)")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.text(0.97, 0.04,
            "Each row's phoneme-token edit distance / target phoneme count.\n"
            "CER ≥ PER is expected: characters strictly refine phoneme tokens.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#555555")
    fig.tight_layout()
    fig.savefig(FIG / "fig06_per_vs_cer.png")
    plt.close(fig)
    print("wrote fig06_per_vs_cer.png")


if __name__ == "__main__":
    fig01_beam5_vs_greedy()
    fig02_sid_stacked()
    fig03_failure_modes()
    fig04_real_word_breakdown()
    fig05_oov_target_vs_pred()
    fig06_per_vs_cer()
    print(f"all figures written to {FIG}")
