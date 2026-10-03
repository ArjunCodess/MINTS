"""Extract final PDF text, metadata and annotations for submission screening.

Run using a Python environment with pypdf (the bundled document runtime works).
"""
from pathlib import Path
import hashlib
import json
from pypdf import PdfReader

root=Path(__file__).resolve().parents[1]
pdf=root/"paper/main.pdf"
reader=PdfReader(pdf)
text="\n".join(page.extract_text() or "" for page in reader.pages)
annotations=[]
for index,page in enumerate(reader.pages):
    for reference in page.get("/Annots",[]):
        item=reference.get_object()
        annotations.append(dict(page=index+1,subtype=str(item.get("/Subtype","")),
                                contents=str(item.get("/Contents","")),title=str(item.get("/T","")),
                                uri=str(item.get("/A",{}).get("/URI",""))))
report=dict(pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),pages=len(reader.pages),
            metadata={str(k):str(v) for k,v in (reader.metadata or {}).items()},annotations=annotations,text=text)
(root/"results/review/pdf_inspection.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k!="text"},indent=2))
