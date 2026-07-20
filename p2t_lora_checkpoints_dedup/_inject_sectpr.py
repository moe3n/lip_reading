"""Inject a default w:sectPr at end of body if missing.

python-docx refuses to add tables unless at least one w:sectPr exists
in the body. The intermediate findings.docx was left without one
because the appender scripts saved partial state. This script
injects a Letter-size, 1in-margin sectPr just before </w:body>.
"""
import os
import shutil
import zipfile

DOC = r"p2t_lora_checkpoints_dedup\analysis\findings.docx"
SECTPR = (
    '<w:sectPr>'
    '<w:pgSz w:w="12240" w:h="15840"/>'
    '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"'
    ' w:header="720" w:footer="720" w:gutter="0"/>'
    '<w:cols w:space="720"/>'
    '<w:docGrid w:linePitch="360"/>'
    '</w:sectPr>'
)

tmp = DOC + ".tmp"
shutil.copyfile(DOC, tmp)
zin = zipfile.ZipFile(tmp, "r")
zout = zipfile.ZipFile(DOC, "w", zipfile.ZIP_DEFLATED)
for item in zin.namelist():
    data = zin.read(item)
    if item == "word/document.xml":
        xml = data.decode("utf-8")
        if "<w:sectPr" not in xml:
            xml = xml.replace("</w:body>", SECTPR + "</w:body>")
        data = xml.encode("utf-8")
    zout.writestr(item, data)
zin.close()
zout.close()
os.remove(tmp)
print("sectPr injected; size:", os.path.getsize(DOC))
