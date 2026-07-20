"""Render fig07 / fig08 — 10x10 phoneme confusion heatmaps.

Inputs:
  analysis/tables/phoneme_confusion_all949.csv
  analysis/tables/phoneme_confusion_emfalse.csv

Selection: top-10 ref_phon by substitution mass (sum of `count`
across hyp_phon), ties broken by ref_count (occurrences in target
streams), then alphabetical.

Outputs:
  analysis/figures/fig07_phoneme_confusion_all949.png
  analysis/figures/fig08_phoneme_confusion_emfalse.png
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

CSV_ALL = TABLES / "phoneme_confusion_all949.csv"
CSV_EMF = TABLES / "phoneme_confusion_emfalse.csv"


def load_long_form(path):
    """Return pair_counts dict {(ref, hyp): n}, ref_counts dict {ref: n}."""
    pair_counts = {}
    ref_counts = {}
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rp, hp, n = r["ref_phon"], r["hyp_phon"], int(r["count"])
            pair_counts[(rp, hp)] = n
            ref_counts[rp] = int(r["ref_count"])
    return pair_counts, ref_counts


def pick_top_10(pair_counts, ref_counts):
    """Rank ref_phon by substitution mass (sum over hyp), tiebreak by
    ref_count then alphabetical. Return list of 10 ref phonemes.
    """
    mass = {}
    for (rp, _), n in pair_counts.items():
        mass[rp] = mass.get(rp, 0) + n
    ranked = sorted(
        mass.items(),
        key=lambda kv: (-kv[1], -ref_counts.get(kv[0], 0), kv[0]),
    )
    return [rp for rp, _ in ranked[:10]]


def render_heatmap(pair_counts, ref_counts, top10, out_path, title, total_subs):
    # 10x10 matrix: rows = ref, cols = hyp
    matrix = np.zeros((10, 10), dtype=int)
    for (rp, hp), n in pair_counts.items():
        if rp in top10 and hp in top10:
            i = top10.index(rp)
            j = top10.index(hp)
            matrix[i, j] = n
    # Diagonal (ref==hyp) would be EM-True, so will be 0; mask for visual clarity.
    masked = np.ma.masked_where(np.eye(10, dtype=bool), matrix)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(masked, cmap=cmap, aspect="equal", vmin=0, vmax=max(1, matrix.max()))

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(top10, fontsize=11)
    ax.set_yticklabels(top10, fontsize=11)
    ax.set_xlabel("predicted (hyp) phoneme", fontsize=12)
    ax.set_ylabel("target (ref) phoneme", fontsize=12)
    ax.set_title(
        f"{title}\n"
        f"Top-10 by substitution mass · {int(matrix.sum())} sub events shown",
        fontsize=13,
    )

    # annotate every cell; show "—" for diagonal (would be EM-True, never observed)
    for i in range(10):
        for j in range(10):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        color="#999999", fontsize=11)
            else:
                v = int(matrix[i, j])
                color = "white" if v >= matrix.max() * 0.6 and v > 0 else "black"
                ax.text(j, i, str(v) if v else "", ha="center", va="center",
                        color=color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("# substitution events", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return matrix


def main():
    # all-949
    pc_all, rc_all = load_long_form(CSV_ALL)
    top10_all = pick_top_10(pc_all, rc_all)
    m_all = render_heatmap(
        pc_all, rc_all, top10_all,
        FIG / "fig07_phoneme_confusion_all949.png",
        "Figure 7 — Phoneme confusion (all-949)",
        total_subs=sum(pc_all.values()),
    )
    print(f"[fig07] top10 (ref axis, ranked by sub mass): {top10_all}")
    print(f"[fig07] non-zero off-diagonal cells: {(m_all - np.eye(10, dtype=int) * m_all.diagonal()).sum() - 0}")
    nz = [(top10_all[i], top10_all[j], int(m_all[i, j]))
          for i in range(10) for j in range(10)
          if i != j and m_all[i, j] > 0]
    nz.sort(key=lambda x: (-x[2], x[0], x[1]))
    print(f"[fig07] non-zero confusions: {nz}")

    # em-false-76
    pc_emf, rc_emf = load_long_form(CSV_EMF)
    top10_emf = pick_top_10(pc_emf, rc_emf)
    m_emf = render_heatmap(
        pc_emf, rc_emf, top10_emf,
        FIG / "fig08_phoneme_confusion_emfalse.png",
        "Figure 8 — Phoneme confusion (EM-False 76)",
        total_subs=sum(pc_emf.values()),
    )
    print(f"[fig08] top10 (ref axis, ranked by sub mass): {top10_emf}")
    nz = [(top10_emf[i], top10_emf[j], int(m_emf[i, j]))
          for i in range(10) for j in range(10)
          if i != j and m_emf[i, j] > 0]
    nz.sort(key=lambda x: (-x[2], x[0], x[1]))
    print(f"[fig08] non-zero confusions: {nz}")


if __name__ == "__main__":
    main()