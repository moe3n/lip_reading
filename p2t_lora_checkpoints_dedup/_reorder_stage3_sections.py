"""One-off: move the appended Sections 7-10 (currently at end of doc) to be
inserted BEFORE the Appendix heading. Done via XML element relocation.
"""

from docx import Document

DOC = r"p2t_lora_checkpoints_dedup\analysis\findings.docx"

d = Document(DOC)

paragraphs = list(d.paragraphs)
appendix_elem = None
section7_elem = None

for p in paragraphs:
    if p.text.startswith("Appendix"):
        appendix_elem = p._element
    elif p.text.startswith("7. Weighted") and section7_elem is None:
        section7_elem = p._element

if appendix_elem is None or section7_elem is None:
    raise SystemExit("could not locate boundary elements")

# Collect every XML element from section7_elem (inclusive) to end of body
moveable = []
cur = section7_elem
while cur is not None:
    nxt = cur.getnext()
    moveable.append(cur)
    cur = nxt

print(f"moving {len(moveable)} elements")

# Remove them from current location (in reverse to keep siblings stable)
parent = section7_elem.getparent()
for elem in moveable:
    parent.remove(elem)

# Insert them BEFORE the Appendix element
for elem in moveable:
    appendix_elem.addprevious(elem)

d.save(DOC)
print("reordered and saved")