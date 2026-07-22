"""Figures for the noise-augmented training report.

Reads the two robustness-probe summaries and the two training histories, writes
two figures:

  fig_robustness_curves.png  exact match vs corruption rate, one panel per
                             corruption type, clean-trained against noise-trained
  fig_training_loss.png      train and validation loss per epoch, both models

No model or GPU needed; reads CSV/JSON already on disk.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
CLEAN_PROBE = ROOT / "analysis" / "noise_probe_p2t_lora_checkpoints_dedup" / "summary.csv"
NOISE_PROBE = ROOT / "analysis" / "noise_probe_p2t_lora_checkpoints_noise" / "summary.csv"
CLEAN_HIST = ROOT / "p2t_lora_checkpoints_dedup" / "training_history.json"
NOISE_HIST = ROOT / "p2t_lora_checkpoints_noise" / "training_history.json"
OUT = ROOT / "analysis" / "figures_noise"
OUT.mkdir(parents=True, exist_ok=True)

CLEAN_COLOR = "#c0392b"   # clean-trained
NOISE_COLOR = "#1d6fb8"   # noise-trained


def load_probe(path):
    """Return {kind: {rate: EM}} plus the clean-control EM."""
    by_kind, clean_em = {}, None
    with path.open() as f:
        for r in csv.DictReader(f):
            em = float(r["ExactMatch"])
            if r["kind"] == "clean":
                clean_em = em
            else:
                by_kind.setdefault(r["kind"], {})[float(r["rate"])] = em
    return by_kind, clean_em


def robustness_figure():
    clean, clean_ctrl = load_probe(CLEAN_PROBE)
    noise, noise_ctrl = load_probe(NOISE_PROBE)
    kinds = ["substitute", "delete", "insert"]
    titles = {"substitute": "Substitution", "delete": "Deletion", "insert": "Insertion"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, kind in zip(axes, kinds):
        # x = 0 is the clean control, then 5/10/20%.
        rates = [0.0] + sorted(clean[kind])
        clean_y = [clean_ctrl] + [clean[kind][r] for r in sorted(clean[kind])]
        noise_y = [noise_ctrl] + [noise[kind][r] for r in sorted(noise[kind])]
        xs = [r * 100 for r in rates]

        ax.plot(xs, clean_y, "o-", color=CLEAN_COLOR, label="Clean-trained", lw=2)
        ax.plot(xs, noise_y, "s-", color=NOISE_COLOR, label="Noise-trained", lw=2)
        ax.set_title(titles[kind], fontsize=11)
        ax.set_xlabel("Input phonemes corrupted (%)", fontsize=9)
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    axes[0].set_ylabel("Exact sentence match (%)", fontsize=10)
    axes[0].legend(fontsize=9, loc="upper right")
    fig.suptitle("Robustness to phoneme-input corruption (300 sentences, beam-5)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig_robustness_curves.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'fig_robustness_curves.png'}")


def loss_figure():
    clean = json.loads(CLEAN_HIST.read_text())
    noise = json.loads(NOISE_HIST.read_text())
    ep = [e["epoch"] for e in clean]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(ep, [e["train_loss"] for e in clean], "o-", color=CLEAN_COLOR,
            label="Clean-trained, training")
    ax.plot(ep, [e["val_loss"] for e in clean], "o--", color=CLEAN_COLOR,
            label="Clean-trained, validation", alpha=0.7)
    ax.plot(ep, [e["train_loss"] for e in noise], "s-", color=NOISE_COLOR,
            label="Noise-trained, training")
    ax.plot(ep, [e["val_loss"] for e in noise], "s--", color=NOISE_COLOR,
            label="Noise-trained, validation", alpha=0.7)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_xticks(ep)
    ax.set_title("Training and validation loss per epoch", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_training_loss.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'fig_training_loss.png'}")


if __name__ == "__main__":
    robustness_figure()
    loss_figure()
