$here = @'
"""Rebuild Direct_Baseline_Colab.ipynb with attention-enabled seq2seq."""
import json
from pathlib import Path

NB_PATH = Path(r'c:\Projects\lip_reading\notebooks\Direct_Baseline_Colab.ipynb')

def L(*ls): return list(ls)
def md(*ls): return {'cell_type':'markdown','metadata':{},'source':L(*ls)}
def co(*ls): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':L(*ls)}

cells = []

# === Title ===
cells.append(md(
    '# Direct Phoneme -> Text Baseline (no LLM, no context)',
    '',
    'A minimal **GRU encoder-decoder + Bahdanau attention** trained from scratch on the LRS2 phoneme->text corpus. Purpose: establish the *floor* for the thesis comparison.',
    '',
    'Trains in ~15-20 min on T4 or ~60 min on Colab CPU. Designed for the free Colab tier.',
    '',
    '**v2 note:** the original GRU decoder without attention collapsed into 5-char cycles. This version adds Bahdanau cross-attention so the decoder can look at every encoder time-step.',
))

# === Mount Drive ===
cells.append(md('## 1. Setup -- mount Drive and confirm data'))
cells.append(co(
    'from google.colab import drive',
    'drive.mount(\'/content/drive\')',
    '',
    'DATA_DIR = \'/content/drive/MyDrive/P2T/data\'',
    'CORPUS_CSV = f\'{DATA_DIR}/sentphonemepairs_LRS2_original.csv\'',
    '',
    'import os',
    'assert os.path.isfile(CORPUS_CSV), (',
    '    f\'Could not find {CORPUS_CSV}. \',',
    '    f\'Files in {DATA_DIR}: {os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else chr(60)+\"dir missing\"+chr(62)}\',',
    ')',
    'print(f\'Using corpus: {CORPUS_CSV}\')',
    'print(f\'Corpus size: {os.path.getsize(CORPUS_CSV) / 1e6:.2f} MB\')',
    '',
    'OUT_DIR = \'/content/drive/MyDrive/P2T/direct_baseline_out\'',
    'os.makedirs(OUT_DIR, exist_ok=True)',
    'print(f\'Output dir: {OUT_DIR}\')',
))

# === Deps ===
cells.append(md(
    '## 2. Install/check dependencies (preinstalled on Colab)',
    '',
    'torch / numpy are preinstalled. We use only standard library + these -- no transformers, no peft, no bitsandbytes.',
))
cells.append(co(
    'import torch',
    'print(f\'torch={torch.__version__}  cuda={torch.cuda.is_available()}\')',
    'DEVICE = torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')',
    'print(f\'Device: {DEVICE}\')',
))

# === Hyperparameters ===
cells.append(md(
    '## 3. Hyperparameters',
    '',
    'Small enough to train end-to-end on Colab free tier within an hour.',
))
cells.append(co(
    '# Default (fits Colab free tier, ~15-20 min on T4)',
    'N_PAIRS        = 5000     # 0 = full 48k',
    'MAX_TRAIN      = 4000     # absolute cap on train rows after 80/20 split',
    'EMB_DIM        = 64',
    'HID_DIM        = 128',
    'N_LAYERS       = 1',
    'DROPOUT        = 0.2',
    'BATCH_SIZE     = 32',
    'N_EPOCHS       = 8',
    'LR             = 3e-3',
    'TEACHER_FORCE_P = 0.5',
    'MAX_TEXT_LEN   = 80',
    'PHONEME_VOCAB_MAX = 200',
    'TEXT_VOCAB_MAX    = 200',
    '',
    '# Scale-up (T4, ~1-2 h): uncomment',
    '# N_PAIRS, MAX_TRAIN = 0, 30000',
    '# EMB_DIM, HID_DIM, BATCH_SIZE = 128, 256, 64',
    '# N_EPOCHS = 12',
))

# Save head and stop (model + train cells appended next).
NB_PATH.write_text(json.dumps(
    {'cells': cells,
     'metadata': {
         'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
         'language_info': {'name':'python','version':'3.10'},
         'colab': {'provenance': [], 'gpuType': 'T4'},
     },
     'nbformat': 4, 'nbformat_minor': 0},
    indent=1, ensure_ascii=False), encoding='utf-8')
print('head saved, cells=', len(cells))
'@

Set-Content -Path c:\Projects\lip_reading\rebuild_nb.py -Value $here -Encoding UTF8 -NoNewline
python c:\Projects\lip_reading\rebuild_nb.py