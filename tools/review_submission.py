"""Check generated numeric inserts and scan the manuscript package for remnants."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import re
import pandas as pd
from tools.review_tables import tex_table

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/review"
FORBIDDEN=[r"Response.to.Review",r"review.triage",r"response.checklist",r"mV6D",r"C5jL",r"ndFG",r"preregistered",r"pre-registered",
           r"registered threshold",r"Arjun",r"prakash12345",r"Kiho also",r"GPT-5\.4",r"GPT-5\.5",r"ICML 2026",r"rebuttal"]


def stale_findings(text):
    return [pattern for pattern in FORBIDDEN if re.search(pattern,text,re.I)]


def validate_tables():
    metrics=pd.read_csv(OUT/"classification_metrics.csv")
    for filename,methods in [("table3.tex",dict(gc="GC",kmer="$3$--$6$-mer",probe="Frozen readout")),
                             ("table4.tex",dict(probe="Full",historical_gc_probe="Historical matching",caliper_gc_probe="GC caliper"))]:
        if (OUT/filename).read_text()!=tex_table(metrics,methods):
            raise ValueError(f"Stale generated table: {filename}")
    for metric in ["auroc","auprc","accuracy"]:
        if not metrics[metric].between(0,1).all():
            raise ValueError(f"Out-of-range {metric}")


def main():
    validate_tables()
    package=[ROOT/"paper/main.tex",ROOT/"paper/references.bib",*OUT.glob("*.tex")]
    findings={str(p.relative_to(ROOT)):stale_findings(p.read_text(encoding="utf-8")) for p in package}
    findings={p:v for p,v in findings.items() if v}
    if findings:
        raise ValueError(f"Submission remnants: {findings}")
    report=dict(submission_text_files=[str(p.relative_to(ROOT)) for p in package], findings=findings,
                scope="rendered manuscript sources and inserts; internal review documents and historical results are not submission material",
                hash_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in package})
    pdf=ROOT/"paper/main.pdf"
    if pdf.exists():
        report["pdf_sha256"]=hashlib.sha256(pdf.read_bytes()).hexdigest()
        inspection=json.loads((OUT/"pdf_inspection.json").read_text(encoding="utf-8"))
        if inspection["pdf_sha256"]!=report["pdf_sha256"]:
            raise ValueError("Stale PDF inspection; rerun tools/review_pdf.py")
        pdf_findings=stale_findings(inspection["text"]+json.dumps(inspection["metadata"])+json.dumps(inspection["annotations"]))
        if pdf_findings:
            raise ValueError(f"PDF remnants: {pdf_findings}")
        if re.search(r"Table\s+[89]\b",inspection["text"]):
            raise ValueError("Stale table 8 or 9 in PDF")
        report["pdf_pages"]=inspection["pages"]
        report["pdf_metadata_and_annotations_scanned"]=True
    (OUT/"submission_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
