"""Overview figure of every full-corpus run and its headline results.

Numbers are read from the metrics files on disk (see comments), not re-simulated.
Two panels: exact sentence match (higher is better) and word error rate (lower
is better), five runs grouped by approach.

Writes analysis/figures_overview/fig_runs_overview.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "figures_overview"
OUT.mkdir(parents=True, exist_ok=True)

# (label, approach, WER%, EM%, note)
# GRU:        direct_baseline_out/direct_baseline_metrics.csv  (1000 val, greedy)
# Zero-shot:  computed from zero-shot/baseline/preds_train_45839_clean.jsonl
# LoRA full:  p2t_lora_checkpoints_full/metrics_log.csv        (1082 val, beam-5, leaked)
# LoRA dedup: p2t_lora_checkpoints_dedup/metrics_beam5.csv     (949 dedup val, beam-5)
# LoRA noise: p2t_lora_checkpoints_noise/metrics_log.csv       (949 dedup val, beam-5)
RUNS = [
    ("GRU\nbaseline", "baseline", 91.23, 0.00, ""),
    ("Zero-shot\nLlama 3.2 3B", "prompt", 127.78, 0.21, ""),
    ("LoRA\ncontaminated", "lora_flag", 1.81, 93.53, "superseded"),
    ("LoRA\ndeduplicated", "lora", 2.09, 91.99, ""),
    ("LoRA\nnoise-aug.", "lora", 2.92, 88.73, ""),
]

COLORS = {"baseline": "#7f8c8d", "prompt": "#e08a1e",
          "lora": "#1d6fb8", "lora_flag": "#9db8d2"}


def bar_panel(ax, values, title, ylabel, fmt):
    x = np.arange(len(RUNS))
    for i, (label, approach, wer, em, note) in enumerate(RUNS):
        hatch = "//" if approach == "lora_flag" else None
        ax.bar(i, values[i], color=COLORS[approach], hatch=hatch,
               edgecolor="white", linewidth=0.5)
        ax.text(i, values[i] + max(values) * 0.015, fmt.format(values[i]),
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in RUNS], fontsize=8, rotation=0)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    bar_panel(ax1, [r[3] for r in RUNS],
              "Exact sentence match (higher is better)", "Exact match (%)", "{:.1f}%")
    ax1.set_ylim(0, 100)

    bar_panel(ax2, [r[2] for r in RUNS],
              "Word error rate (lower is better)", "WER (%)", "{:.1f}%")
    ax2.set_ylim(0, 140)

    # Shared legend describing the approaches.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["baseline"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["prompt"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["lora"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["lora_flag"], hatch="//"),
    ]
    fig.legend(handles,
               ["No language model", "Prompting only", "LoRA fine-tuned",
                "LoRA (contaminated eval, superseded)"],
               loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Full-corpus runs: phoneme-to-text on LRS2", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(OUT / "fig_runs_overview.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'fig_runs_overview.png'}")


if __name__ == "__main__":
    main()
