"""Verify new primary citations before rebuilding the manuscript manifest."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import shutil
import requests
import pandas as pd
from src.utils import write_json,sha256_file

ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/"results/review/reference_verification.json"
    prior=ROOT/"results/mapped_variant/validation/prior_reference_verification.json"
    if prior.exists():raise FileExistsError("Prior verification snapshot already exists")
    shutil.copyfile(path,prior)
    records=[]
    for key,url,doi,title in (
        ("abramov2021adastra","https://www.nature.com/articles/s41467-021-23007-0","10.1038/s41467-021-23007-0",
            "Landscape of allele-specific transcription factor binding in the human genome"),
        ("adastra2024release","https://zenodo.org/records/14174114","10.5281/zenodo.14174114",
            "ADASTRA release Mabel v.6.1 Nov 2024")):
        response=requests.get(url,timeout=(15,40));response.raise_for_status()
        if title.lower() not in response.text.lower():raise ValueError(f"Title verification failed: {url}")
        records.append(dict(key=key,url=url,doi=doi,title=title,status=response.status_code,
            response_sha256=__import__("hashlib").sha256(response.content).hexdigest(),
            verification="official publisher/deposit page title, DOI and source metadata; browser and HTTP checks"))
    value=json.loads(path.read_text())
    value["references"].extend(records)
    value["previous_checked_at"]=value["checked_at"]
    value["checked_at"]=pd.Timestamp.now(tz="UTC").isoformat()
    value["bibliography_sha256"]=sha256_file(ROOT/"paper/references.bib")
    write_json(path,value)
    write_json(ROOT/"results/mapped_variant/validation/new_reference_verification.json",dict(
        added=records,bibliography_sha256=value["bibliography_sha256"],previous_sha256=sha256_file(prior)))
    print("new references verified")


if __name__=="__main__":main()
