"""Direct Phoneme-to-Text Baseline (NO LLM, NO context)"""

import argparse
import csv
import os
import random
import re
import time
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_CSV = os.path.join(HERE, "src", "cpt_decoder", "data",
                          "sentphonemepairs_LRS2_original.csv")
OUT_DIR = os.path.join(HERE, "direct_baseline_out")

PHONEME_VOCAB_MAX = 200
TEXT_VOCAB_MAX    = 200
EMB_DIM           = 64
HID_DIM           = 128
N_LAYERS          = 1
DROPOUT           = 0.2
BATCH_SIZE        = 32
N_EPOCHS          = 8
LR                = 3e-3
TEACHER_FORCE_P   = 0.5
MAX_TEXT_LEN      = 80
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD, BOS, EOS, UNK = 0, 1, 2, 3

def clean_phonemes(raw: str) -> str:
    """Strip <SOS>/<EOS>/<space>/stress, collapse whitespace."""
    raw = (raw or "").strip()
    raw = re.sub(r"<SOS>|<EOS>", "", raw)
    raw = re.sub(r"[012]", "", raw)
    raw = raw.replace("<space>", " ")
    return re.sub(r"\s+", " ", raw).strip()

def clean_text(raw: str) -> str:
    """Lowercase, keep only [a-z '] + space (no punctuation)."""
    raw = (raw or "").lower()
    raw = re.sub(r"[^a-z ']+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()

def encode_phonemes(s: str) -> List[int]:
    return [phoneme_vocab.get(t, phoneme_vocab["<UNK>"]) for t in s.split()]

def encode_text(s: str) -> List[int]:
    return [text_vocab.get(c, text_vocab["<UNK>"]) for c in s]

class PhonemeTextDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.src = [encode_phonemes(p) for p, _ in pairs]
        self.tgt_in, self.tgt_out = [], []
        for _, t in pairs:
            ids = encode_text(t)
            ids = ids[: MAX_TEXT_LEN - 2]
            self.tgt_in.append([BOS] + ids)
            self.tgt_out.append(ids + [EOS])

    def __len__(self): return len(self.src)
    def __getitem__(self, i): return self.src[i], self.tgt_in[i], self.tgt_out[i]

def collate(batch):
    srcs, tgts_in, tgts_out = zip(*batch)
    def pad(seqs, pad_id=PAD):
        m = max(len(s) for s in seqs)
        return torch.tensor([s + [pad_id] * (m - len(s)) for s in seqs], dtype=torch.long)
    return pad(srcs), pad(tgts_in), pad(tgts_out), torch.tensor([len(s) for s in srcs])

class Encoder(nn.Module):
    def __init__(self, vocab: int, emb: int, hid: int, n_layers: int, drop: float):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.rnn = nn.GRU(emb, hid, n_layers, dropout=drop if n_layers > 1 else 0.0,
                          batch_first=True, bidirectional=False)

    def forward(self, src):
        emb = self.emb(src)
        out, h = self.rnn(emb)
        return out, h.squeeze(0)

class Decoder(nn.Module):
    def __init__(self, vocab: int, emb: int, hid: int, n_layers: int, drop: float):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.rnn = nn.GRU(emb, hid, n_layers, dropout=drop if n_layers > 1 else 0.0,
                          batch_first=True)
        self.fc = nn.Linear(hid, vocab)

    def forward(self, tgt_in, h):
        emb = self.emb(tgt_in)
        out, _ = self.rnn(emb, h.unsqueeze(0))
        return self.fc(out)

class Seq2Seq(nn.Module):
    def __init__(self, src_v: int, tgt_v: int):
        super().__init__()
        self.enc = Encoder(src_v, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
        self.dec = Decoder(tgt_v, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)

    def forward(self, src, tgt_in):
        _, h = self.enc(src)
        return self.dec(tgt_in, h)

@torch.no_grad()
def greedy_decode(model, src, max_len=MAX_TEXT_LEN):
    model.eval()
    src = src.unsqueeze(0).to(DEVICE)
    _, h = model.enc(src)
    y = torch.tensor([[BOS]], device=DEVICE)
    out_ids = []
    for _ in range(max_len):
        logits = model.dec(y, h)[:, -1, :]
        nxt = int(logits.argmax(-1).item())
        if nxt == EOS:
            break
        out_ids.append(nxt)
        y = torch.cat([y, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    return out_ids

def detok(ids: List[int]) -> str:
    return "".join(text_inv.get(i, "?") for i in ids)

def edit_distance(a: List[str], b: List[str]) -> Tuple[int, int]:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[n][m], n

def evaluate(model, ds: PhonemeTextDataset) -> dict:
    model.eval()
    wer_n, wer_d = 0, 0
    cer_n, cer_d = 0, 0
    em_ok = 0
    n = len(ds)
    sample_idx = list(range(n)) if n <= 2000 else random.sample(range(n), 500)
    for i in sample_idx:
        src, _, _ = ds[i]
        ids = greedy_decode(model, torch.tensor(src))
        pred = detok(ids)
        ref = detok(ds.tgt_out[i][:-1])
        wer_d_i, wer_n_i = edit_distance(ref.split(), pred.split())
        wer_d += wer_d_i; wer_n += max(1, wer_n_i)
        cer_d_i, cer_n_i = edit_distance(list(ref), list(pred))
        cer_d += cer_d_i; cer_n += max(1, cer_n_i)
        if pred.strip() == ref.strip():
            em_ok += 1
    return {"n": len(sample_idx),
            "WER": wer_d / wer_n if wer_n else 0.0,
            "CER": cer_d / cer_n if cer_n else 0.0,
            "EM":  em_ok / len(sample_idx)}

def train_one_epoch(model, opt, loader):
    model.train()
    total, n = 0.0, 0
    for src, tgt_in, tgt_out, _ in loader:
        src, tgt_in, tgt_out = src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)
        if random.random() < TEACHER_FORCE_P:
            inp = tgt_in
        else:
            with torch.no_grad():
                _, h = model.enc(src)
                preds = []
                y = tgt_in[:, :1]
                for t in range(tgt_in.size(1) - 1):
                    logits = model.dec(y, h)[:, -1, :]
                    nxt = logits.argmax(-1, keepdim=True)
                    preds.append(nxt)
                    y = torch.cat([y, nxt], dim=1)
                inp = torch.cat([tgt_in[:, :1]] + [p.unsqueeze(1) if p.dim()==0 else p for p in preds], dim=1)
                if inp.size(1) != tgt_in.size(1):
                    inp = tgt_in
        logits = model(src, inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tgt_out.reshape(-1), ignore_index=PAD)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * src.size(0); n += src.size(0)
    return total / n

def load_pairs(n: int, official: bool = False) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(CORPUS_CSV, "r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        for sent, phon in rdr:
            if not sent or not phon:
                continue
            cs = clean_text(sent); cp = clean_phonemes(phon)
            if cs and cp:
                rows.append((cp, cs))
            if n and not official and len(rows) >= n:
                break
    if official:
        TRAIN_N, VAL_N, TEST_N = 45839, 1082, 1243
        return rows[:TRAIN_N], rows[TRAIN_N + VAL_N: TRAIN_N + VAL_N + TEST_N]
    cut = int(len(rows) * 0.8)
    return rows[:cut], rows[cut:]

def build_vocabs(train_pairs: List[Tuple[str, str]]):
    global phoneme_vocab, text_vocab, phoneme_inv, text_inv
    phonemes = Counter(p for p, _ in train_pairs for p in p.split())
    texts    = Counter(c for _, t in train_pairs for c in t)
    phoneme_vocab = {"<PAD>": PAD, "<BOS>": BOS, "<EOS>": EOS, "<UNK>": UNK}
    for p, _ in phonemes.most_common(PHONEME_VOCAB_MAX - 4):
        phoneme_vocab.setdefault(p, len(phoneme_vocab))
    text_vocab = {"<PAD>": PAD, "<BOS>": BOS, "<EOS>": EOS, "<UNK>": UNK, " ": 4}
    for c, _ in texts.most_common(TEXT_VOCAB_MAX - 5):
        if c == " ":
            continue
        text_vocab.setdefault(c, len(text_vocab))
    phoneme_inv = {v: k for k, v in phoneme_vocab.items()}
    text_inv    = {v: k for k, v in text_vocab.items()}
    print(f"  Vocab: phonemes={len(phoneme_vocab)}  text_chars={len(text_vocab)}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000, help="first N pairs (0 = all 48k)")
    p.add_argument("--epochs", type=int, default=N_EPOCHS)
    p.add_argument("--batch", type=int, default=BATCH_SIZE)
    p.add_argument("--max_train", type=int, default=4000,
                   help="absolute cap on train set after 80/20 split (ignored with --official)")
    p.add_argument("--official", action="store_true",
                   help="train on the full 45,839-row LRS2 train split and evaluate on the "
                        "held-out 1,243-row test split, matching the LoRA runs")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60)
    print(f"  Direct Phoneme→Text Baseline  (no LLM, no context)")
    print("=" * 60)
    print(f"  Device : {DEVICE}")
    print(f"  Pairs  : {args.n if args.n else 'all'}")
    print(f"  Epochs : {args.epochs}  Batch: {args.batch}")
    print()

    t0 = time.time()
    print("Loading + cleaning CSV...")
    if args.official:
        train_pairs, val_pairs = load_pairs(0, official=True)
        print(f"  Official split: train={len(train_pairs):,}  test={len(val_pairs):,}")
    else:
        train_pairs, val_pairs = load_pairs(args.n)
        train_pairs = train_pairs[: args.max_train]
        print(f"  Loaded train={len(train_pairs):,}  val={len(val_pairs):,}")
    build_vocabs(train_pairs)

    train_ds = PhonemeTextDataset(train_pairs)
    val_ds   = PhonemeTextDataset(val_pairs)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate, num_workers=0)

    model = Seq2Seq(len(phoneme_vocab), len(text_vocab)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: GRU encoder-decoder, {n_params:,} parameters")
    print(f"  enc: GRU({EMB_DIM}→{HID_DIM}, {N_LAYERS} layer)"
          f"    dec: GRU({EMB_DIM}→{HID_DIM}, {N_LAYERS} layer)\n")

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"{'epoch':>5} {'train_loss':>12} {'val_WER':>9} {'val_CER':>9} {'val_EM':>9}  {'time':>7}")
    print("-" * 60)
    for epoch in range(1, args.epochs + 1):
        ep_t = time.time()
        loss = train_one_epoch(model, opt, train_loader)
        m = evaluate(model, val_ds)
        print(f"{epoch:>5} {loss:>12.4f} {m['WER']*100:>8.2f}% {m['CER']*100:>8.2f}%"
              f" {m['EM']*100:>8.2f}%  {time.time()-ep_t:>6.1f}s")
    print("-" * 60)

    print(f"\nTotal: {time.time()-t0:.1f}s on {DEVICE}")
    print("\nFinal examples (val):")
    for i in random.sample(range(len(val_ds)), min(8, len(val_ds))):
        src, _, _ = val_ds[i]
        ids = greedy_decode(model, torch.tensor(src))
        print(f"  phon: {' '.join(phoneme_inv.get(x, '?') for x in src)}")
        print(f"  ref : {detok(val_ds.tgt_out[i][:-1])!r}")
        print(f"  pred: {detok(ids)!r}")
        print()

    summary_path = os.path.join(OUT_DIR, "direct_baseline_metrics.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_val", "WER", "CER", "EM", "n_params", "device"])
        m = evaluate(model, val_ds)
        w.writerow(["GRU-direct", m["n"], f"{m['WER']*100:.4f}",
                    f"{m['CER']*100:.4f}", f"{m['EM']*100:.4f}",
                    n_params, str(DEVICE)])
    print(f"Metrics written to {summary_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
