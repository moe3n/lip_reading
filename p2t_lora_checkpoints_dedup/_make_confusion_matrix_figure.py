"""Render the phoneme substitution data as an actual confusion matrix heatmap.

Reference phonemes on rows, predicted phonemes on columns, cell value = number
of times that substitution occurred across the 76 failing sentences. Phonemes
are ordered vowels-first then consonants, with a divider, so any vowel-to-vowel
clustering is visible as a block in the top-left.

Reads analysis/tables/phoneme_confusion_tokens_emfalse.csv.
Writes analysis/figures/fig_confusion_matrix_phoneme.png.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
CSV_IN = ROOT / "analysis" / "tables" / "phoneme_confusion_tokens_emfalse.csv"
FIG_OUT = ROOT / "analysis" / "figures" / "fig_confusion_matrix_phoneme.png"
FIG_OUT.parent.mkdir(parents=True, exist_ok=True)

VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
          "IH", "IY", "OW", "OY", "UH", "UW"}


def load_pairs():
    pairs = []
    with CSV_IN.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pairs.append((r["ref_phon"], r["hyp_phon"], int(r["count"])))
    return pairs


def order(symbols):
    """Vowels first (alphabetical), then consonants (alphabetical)."""
    v = sorted(s for s in symbols if s in VOWELS)
    c = sorted(s for s in symbols if s not in VOWELS)
    return v, c


def main():
    pairs = load_pairs()
    refs = {p[0] for p in pairs}
    hyps = {p[1] for p in pairs}

    rv, rc = order(refs)
    hv, hc = order(hyps)
    row_labels = rv + rc
    col_labels = hv + hc
    n_row_vowels, n_col_vowels = len(rv), len(hv)

    ri = {s: i for i, s in enumerate(row_labels)}
    ci = {s: i for i, s in enumerate(col_labels)}

    M = np.zeros((len(row_labels), len(col_labels)), dtype=int)
    for ref, hyp, n in pairs:
        M[ri[ref], ci[hyp]] += n

    fig, ax = plt.subplots(figsize=(9, 8))
    masked = np.ma.masked_where(M == 0, M)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#f5f5f5")
    im = ax.imshow(masked, cmap=cmap, vmin=1, vmax=max(2, M.max()), aspect="equal")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=90)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Predicted phoneme", fontsize=10)
    ax.set_ylabel("Reference phoneme", fontsize=10)
    ax.set_title("Phoneme substitution confusion matrix (76 failing sentences)\n"
                 "vowels grouped top-left, counts shown per cell", fontsize=10)

    # Count labels in filled cells.
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if M[i, j] > 0:
                ax.text(j, i, str(M[i, j]), ha="center", va="center",
                        fontsize=8, color="black" if M[i, j] < 2 else "white")

    # Divider lines separating the vowel block from the consonant block.
    ax.axhline(n_row_vowels - 0.5, color="#c0392b", lw=1.2)
    ax.axvline(n_col_vowels - 0.5, color="#c0392b", lw=1.2)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="#dddddd", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("substitution count", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {FIG_OUT}")
    print(f"matrix: {len(row_labels)} ref x {len(col_labels)} hyp, "
          f"{int(M.sum())} substitutions, {int((M > 0).sum())} filled cells")
    # Quick check: the red divider should isolate a vowel-to-vowel block.
    vv = int(M[:n_row_vowels, :n_col_vowels].sum())
    print(f"vowel-to-vowel substitutions (top-left block): {vv}")


if __name__ == "__main__":
    main()
