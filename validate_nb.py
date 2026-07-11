import json, ast
nb = json.load(open(r'c:\Projects\lip_reading\notebooks\Direct_Baseline_Colab.ipynb', encoding='utf-8'))
print('cells:', len(nb['cells']), '  nbformat:', nb['nbformat'])
for i, c in enumerate(nb['cells']):
    src = c.get('source', [])
    if isinstance(src, list):
        src = ''.join(src)
    nlines = src.count(chr(10)) + 1
    label = (src[:60].replace('\n', ' ')) if c['cell_type'] == 'markdown' else f'{nlines} lines'
    print(f'  [{i:2}] {c["cell_type"]:8}  {label}')

# Validate model cell parses
for idx in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
    src = ''.join(nb['cells'][idx]['source']) if nb['cells'][idx]['cell_type'] == 'code' else None
    if src is None:
        continue
    try:
        ast.parse(src)
        print(f'  cell[{idx}]: AST OK')
    except SyntaxError as e:
        print(f'  cell[{idx}]: SYNTAX ERROR at line {e.lineno}: {e.msg}')
        print('    snippet:', src.splitlines()[max(0, e.lineno-1)][:80])