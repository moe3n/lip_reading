$here = @'
"""Rebuild Direct_Baseline_Colab.ipynb from scratch with attention."""
import json
from pathlib import Path

NB_PATH = Path(r'c:\Projects\lip_reading\notebooks\Direct_Baseline_Colab.ipynb')

def L(*ls): return list(ls)
def md(*ls): return {'cell_type':'markdown','metadata':{},'source':L(*ls)}
def co(*ls): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':L(*ls)}

cells = []

cells.append(md(
    '# Direct Phoneme -> Text Baseline (no LLM, no context)',
    '',
    'A minimal **GRU encoder-decoder + Bahdanau attention** trained from scratch on the LRS2 phoneme->text corpus.',
    '',
    'Trains in ~15-20 min on T4 or ~60 min on Colab CPU.',
    '',
    '**v2 note:** the original GRU decoder without attention collapsed into 5-char cycles. This version adds Bahdanau cross-attention.',
))
'@

Set-Content -Path c:\Projects\lip_reading\rebuild_nb.py -Value $here -Encoding UTF8 -NoNewline
Write-Host "wrote $([System.IO.File]::ReadAllText('c:\Projects\lip_reading\rebuild_nb.py').Length) chars"