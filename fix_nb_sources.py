"""
Fix nbformat source lists so each line ends with '\n' (except the last).
Jupyter expects 'line1\n' + 'line2\n' + ... + 'lastline' so that ''.join(source)
produces a properly multiline string. Our builders wrote lines without trailing
newlines, so ast.parse sees one giant line.
"""
import json

NB_PATH = r'c:\Projects\lip_reading\notebooks\Direct_Baseline_Colab.ipynb'

nb = json.load(open(NB_PATH, encoding='utf-8'))
fixed = 0
for c in nb['cells']:
    src = c.get('source', [])
    if not isinstance(src, list) or not src:
        continue
    # Heuristic: if any element doesn't end with '\n', this cell needs fixing.
    needs = any(not s.endswith('\n') for s in src)
    if not needs:
        continue
    new = []
    for i, s in enumerate(src):
        if i < len(src) - 1 and not s.endswith('\n'):
            new.append(s + '\n')
        else:
            new.append(s)
    c['source'] = new
    fixed += 1

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')
print(f'fixed {fixed} cells in {NB_PATH}')