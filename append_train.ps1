$here = @'
"""Append train / eval / save cells to the notebook."""
import json
from pathlib import Path

NB_PATH = Path(r'c:\Projects\lip_reading\notebooks\Direct_Baseline_Colab.ipynb')
nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
cells = nb['cells']

def md(*ls): return {'cell_type':'markdown','metadata':{},'source':list(ls)}
def co(*ls): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':list(ls)}

# --- Data loading markdown + code ---
cells.append(md('## 5. Data loading + 80/20 split'))
cells.append(co(
    'def load_pairs(n):',
    '    rows = []',
    '    with open(CORPUS_CSV, \'r\', encoding=\'utf-8\', newline=\'\') as f:',
    '        for sent, phon in csv.reader(f):',
    '            if not sent or not phon: continue',
    '            cs = clean_text(sent); cp = clean_phonemes(phon)',
    '            if cs and cp: rows.append((cp, cs))',
    '            if n and len(rows) >= n: break',
    '    cut = int(len(rows) * 0.8)',
    '    return rows[:cut], rows[cut:]',
    '',
    'print(\'Loading + cleaning CSV...\')',
    't0 = time.time()',
    'train_pairs, val_pairs = load_pairs(N_PAIRS)',
    'train_pairs = train_pairs[:MAX_TRAIN]',
    'print(f\'  Loaded train={len(train_pairs):,}  val={len(val_pairs):,}  ({time.time()-t0:.1f}s)\')',
    'build_vocabs(train_pairs)',
    '',
    'train_ds = PhonemeTextDataset(train_pairs)',
    'val_ds   = PhonemeTextDataset(val_pairs)',
    'train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=0)',
    'print(f\'\\nDataset: {len(train_ds):,} train / {len(val_ds):,} val\')',
))

# --- Train markdown + code ---
cells.append(md(
    '## 6. Train',
    '',
    'Scheduled-sampling-lite: 50% teacher forcing / 50% model predictions. Adam @ 3e-3, gradient clipping 1.0.',
))
train_lines = [
    'def train_one_epoch(model, opt, loader):',
    '    model.train()',
    '    total, n = 0.0, 0',
    '    for src, tgt_in, tgt_out, src_mask in loader:',
    '        src      = src.to(DEVICE)',
    '        tgt_in   = tgt_in.to(DEVICE)',
    '        tgt_out  = tgt_out.to(DEVICE)',
    '        src_mask = src_mask.to(DEVICE)',
    '        if random.random() < TEACHER_FORCE_P:',
    '            inp = tgt_in',
    '        else:',
    '            # Scheduled sampling: feed previous prediction back.',
    '            with torch.no_grad():',
    '                enc_outs = model.enc(src, src_mask)',
    '                ctx_init = enc_outs.mean(dim=1)',
    '                dec_h    = model.bridge(ctx_init)',
    '                y = tgt_in[:, :1]',
    '                preds = []',
    '                for t in range(tgt_in.size(1) - 1):',
    '                    logits = model.dec(y, dec_h, enc_outs, src_mask)[:, -1, :]',
    '                    nxt    = logits.argmax(-1, keepdim=True)',
    '                    preds.append(nxt)',
    '                    y      = torch.cat([y, nxt], dim=1)',
    '                inp = torch.cat([tgt_in[:, :1]] + preds, dim=1)[:, :tgt_in.size(1)]',
    '        logits = model(src, src_mask, inp)',
    '        loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)),',
    '                                 tgt_out.reshape(-1), ignore_index=PAD)',
    '        opt.zero_grad(); loss.backward()',
    '        nn.utils.clip_grad_norm_(model.parameters(), 1.0)',
    '        opt.step()',
    '        total += loss.item() * src.size(0); n += src.size(0)',
    '    return total / n',
]
train_lines += [
    '',
    'def evaluate(model, ds):',
    '    model.eval()',
    '    wer_d, wer_n = 0, 0',
    '    cer_d, cer_n = 0, 0',
    '    em_ok = 0',
    '    n = len(ds)',
    '    sample_idx = list(range(n)) if n <= 2000 else random.sample(range(n), 500)',
    '    for i in sample_idx:',
    '        src, _, _ = ds[i]',
    '        ids = greedy_decode(model, torch.tensor(src))',
    '        pred = detok(ids)',
    '        ref  = detok(ds.tgt_out[i][:-1])',
    '        ed, nn_ = edit_distance(ref.split(), pred.split())',
    '        wer_d += ed; wer_n += max(1, nn_)',
    '        ed, nn_ = edit_distance(list(ref), list(pred))',
    '        cer_d += ed; cer_n += max(1, nn_)',
    '        if pred.strip() == ref.strip(): em_ok += 1',
    '    return {\'n\': len(sample_idx), \'WER\': wer_d/wer_n if wer_n else 0.0,',
    '            \'CER\': cer_d/cer_n if cer_n else 0.0, \'EM\': em_ok/len(sample_idx)}',
]
train_lines += [
    '',
    'model = Seq2Seq(len(phoneme_vocab), len(text_vocab)).to(DEVICE)',
    'n_params = sum(p.numel() for p in model.parameters())',
    'print(f\'Model: GRU+attn, {n_params:,} parameters\')',
    'opt = torch.optim.Adam(model.parameters(), lr=LR)',
    '',
    'print(f"\\n{\'epoch\':>5} {\'train_loss\':>12} {\'val_WER\':>9} {\'val_CER\':>9} {\'val_EM\':>9}  {\'time\':>7}")',
    'print(\'-\' * 60)',
    'for epoch in range(1, N_EPOCHS + 1):',
    '    ep_t = time.time()',
    '    loss = train_one_epoch(model, opt, train_loader)',
    '    m = evaluate(model, val_ds)',
    '    print(f"{epoch:>5} {loss:>12.4f} {m[\'WER\']*100:>8.2f}% {m[\'CER\']*100:>8.2f}% {m[\'EM\']*100:>8.2f}%  {time.time()-ep_t:>6.1f}s")',
    'print(\'-\' * 60)',
]
cells.append(co(*train_lines))

# --- Examples + summary ---
cells.append(md(
    '## 7. Examples + final metrics',
    '',
    'Sample 8 val rows and write the summary CSV to Drive.',
))
summary_lines = [
    'print(\'\\nFinal examples (val):\')',
    'for i in random.sample(range(len(val_ds)), min(8, len(val_ds))):',
    '    src, _, _ = val_ds[i]',
    '    ids = greedy_decode(model, torch.tensor(src))',
    '    print(f"  phon: {\' \'.join(phoneme_inv.get(x, \'?\') for x in src)}")',
    '    print(f"  ref : {detok(val_ds.tgt_out[i][:-1])!r}")',
    '    print(f"  pred: {detok(ids)!r}")',
    '    print()',
    '',
    'summary_path = os.path.join(OUT_DIR, \'direct_baseline_metrics.csv\')',
    'with open(summary_path, \'w\', newline=\'\') as f:',
    '    w = csv.writer(f)',
    '    w.writerow([\'model\', \'n_val\', \'WER\', \'CER\', \'EM\', \'n_params\', \'device\'])',
    '    m = evaluate(model, val_ds)',
    '    w.writerow([\'GRU-direct-attn\', m[\'n\'], f"{m[\'WER\']*100:.4f}", f"{m[\'CER\']*100:.4f}", f"{m[\'EM\']*100:.4f}", n_params, str(DEVICE)])',
    'print(f\'Metrics written to {summary_path}\')',
]
cells.append(co(*summary_lines))

# --- Done markdown ---
cells.append(md(
    '## Done',
    '',
    '- `direct_baseline_metrics.csv` is in `MyDrive/P2T/direct_baseline_out/`.',
    '- Drop those numbers into the thesis comparison table:',
    '',
    '| Method | WER | CER | EM |',
    '|---|---|---|---|',
    '| Zero-shot Llama-3.2-3B | _(from `zero_shot_baseline.py`)_ | | |',
    '| Direct GRU+attn (this notebook) | _your number_ | _your number_ | _your number_ |',
    '| LoRA CPT decoder | _(thesis number)_ | | |',
    '',
    'Even with attention, expect this row to land in the 60-90% WER range -- still 5-20x worse than zero-shot Llama, which is the point: it quantifies the LLM prior itself.',
))

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('train/eval/save cells appended, total =', len(cells))
'@

Set-Content -Path c:\Projects\lip_reading\rebuild_nb.py -Value $here -Encoding UTF8 -NoNewline
python c:\Projects\lip_reading\rebuild_nb.py