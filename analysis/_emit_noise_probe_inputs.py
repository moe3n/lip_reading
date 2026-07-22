"""Emit the exact corrupted phoneme strings used by analysis/noise_probe.py
without touching the model. Reconstructs byte-identical inputs from the same
corpus, the same seed, the same inventory, and the same per-condition
random.Random() reset.

Run from repo root (so paths to data/ match):

    .venv\\Scripts\\python.exe analysis\\_emit_noise_probe_inputs.py

Writes:

    analysis/noise_probe_out/inputs.csv            one row per probe sentence
    analysis/noise_probe_out/probe_index.csv        which val_clean rows were sampled
    analysis/noise_probe_out/config.json            n_rows, rates, seed, checkpoint
"""
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from p2t_lora.data import loader as data_loader
from p2t_lora.augmentation.phoneme_noise import corrupt, phoneme_inventory

CHECKPOINT_ROOT = ROOT / "p2t_lora_checkpoints_dedup"
EPOCH_DIR = CHECKPOINT_ROOT / "epoch_3"
OUT_DIR = ROOT / "analysis" / "noise_probe_out"
TRAIN_N, VAL_N = 45839, 1082

# Defaults from analysis/noise_probe.py; override with env vars if you changed
# them when running the original probe.
N_ROWS = int(os.environ.get("NOISE_N_ROWS", "300"))
RATES = [float(r) for r in os.environ.get("NOISE_RATES", "0.05,0.10,0.20").split(",")]
SEED = int(os.environ.get("NOISE_SEED", "42"))

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading corpus ...")
full_df = data_loader.load_original_phoneme_text_pairs()
train_df = full_df.iloc[:TRAIN_N].reset_index(drop=True)
val_df = full_df.iloc[TRAIN_N:TRAIN_N + VAL_N].reset_index(drop=True)
val_clean = val_df[~val_df["sentence"].isin(set(train_df["sentence"]))].reset_index(drop=True)

probe = val_clean.sample(n=min(N_ROWS, len(val_clean)), random_state=SEED).reset_index(drop=True)

inventory = phoneme_inventory(full_df["phonemes"])
print(f"Probing {len(probe)} rows | {len(inventory)} phonemes in inventory | seed {SEED}")

# Persist which val_clean indices were chosen so a future model run can verify
# byte-identical alignment against this file.
probe_index = pd.DataFrame({
    "probe_row": range(len(probe)),
    "val_clean_index": probe.index,
    "sentence": probe["sentence"],
    "phonemes": probe["phonemes"],
})
probe_index.to_csv(OUT_DIR / "probe_index.csv", index=False)

# conditions list mirrors noise_probe.py exactly
conditions = [("clean", 0.0)] + [
    (k, r) for k in ("substitute", "delete", "insert") for r in RATES
]

# Pre-build all columns so we can write a single CSV.
columns = {"target": list(probe["sentence"]), "phonemes_clean": list(probe["phonemes"])}

for kind, rate in conditions:
    tag = "clean" if kind == "clean" else f"{kind}_{int(rate * 100)}pct"
    rng = random.Random(SEED)  # one fresh RNG per condition, identical to noise_probe.py
    if kind == "clean":
        columns[f"phonemes_{tag}"] = list(probe["phonemes"])
    else:
        columns[f"phonemes_{tag}"] = [
            corrupt(p, kind, rate, rng, inventory) for p in probe["phonemes"]
        ]
    print(f"  built column phonemes_{tag}")

df = pd.DataFrame(columns)
df.to_csv(OUT_DIR / "inputs.csv", index=False)

with open(OUT_DIR / "config.json", "w") as f:
    json.dump(
        {
            "n_rows": len(probe),
            "rates": RATES,
            "seed": SEED,
            "checkpoint": str(EPOCH_DIR),
            "decoding": "beam-5",
        },
        f,
        indent=2,
    )

print(f"Done. Wrote {OUT_DIR / 'inputs.csv'} ({df.shape}), "
      f"{OUT_DIR / 'probe_index.csv'}, {OUT_DIR / 'config.json'}")
