"""One-off: move the appended Sections 11-13 (currently at end of doc) to be
inserted BEFORE the Appendix heading. Done via XML element relocation.
"""

from docx import Document

DOC = r"p2t_lora_checkpoints_dedup\analysis\findings.docx"

d = Document(DOC)

paragraphs = list(d.paragraphs)
appendix_elem = None
section11_elem = None

for p in paragraphs:
    if p.text.strip().startswith("Appendix"):
        appendix_elem = p._element
    elif p.text.strip().startswith("11. Grammar") and section11_elem is None:
        section11_elem = p._element

if appendix_elem is None or section11_elem is None:
    raise SystemExit("could not locate boundary elements (Appendix or Section 11)")

# Collect every XML element from section11_elem (inclusive) to end of body
moveable = []
cur = section11_elem
while cur is not None:
    nxt = cur.getnext()
    moveable.append(cur)
    cur = nxt

print(f"moving {len(moveable)} elements")

# Remove them from current location (in reverse to keep siblings stable)
parent = section11_elem.getparent()
for elem in moveable:
    parent.remove(elem)

# Insert them BEFORE the Appendix element
for elem in moveable:
    appendix_elem.addprevious(elem)

d.save(DOC)
print("reordered and saved")