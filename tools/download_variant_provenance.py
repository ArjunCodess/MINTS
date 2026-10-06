"""Retrieve small official schema and experiment provenance, preserving failures."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import re
import shutil
import time
import requests
from src.utils import write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/"data/adastra/provenance_v1"
    output.mkdir(parents=True,exist_ok=False)
    receipt=ROOT/"results/variant_provenance"
    receipt.mkdir(parents=True,exist_ok=False)
    attempts=[]
    def get(url,name):
        started=time.time()
        record=dict(url=url,access_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),name=name)
        try:
            response=requests.get(url,timeout=(15,35))
            record.update(http_status=response.status_code,final_url=response.url)
            response.raise_for_status()
            if len(response.content)>25*1024*1024:
                raise ValueError("Small provenance download cap exceeded")
            (output/name).write_bytes(response.content)
            record.update(status="downloaded",bytes=len(response.content),sha256=hashlib.sha256(response.content).hexdigest())
            return response
        except Exception as exc:
            record.update(status="failed",reason=f"{type(exc).__name__}: {exc}")
        finally:
            record["elapsed_seconds"]=time.time()-started
            attempts.append(record)
            write_json(receipt/"downloads.json",dict(disk_free_bytes=shutil.disk_usage(ROOT).free,attempts=attempts))
    readme=get("https://zenodo.org/records/14174114/files/readme.ADASTRA.Mabel.v6.1.txt?download=1","readme.txt")
    if readme:
        actual=hashlib.md5(readme.content).hexdigest()
        if actual!="45e7b826c1b4043bf2b28d10d2e1b134":
            raise ValueError("Publisher readme MD5 mismatch")
        write_json(receipt/"schema.json",dict(publisher_md5=actual,readme_sha256=hashlib.sha256(readme.content).hexdigest(),
            license="CC BY 4.0 deposit; consult source-specific terms for underlying experiments",
            release="Mabel v6.1; live site v6.1.1 is a different version"))
    page=get("https://adastra.autosome.org/mabel/downloads","downloads.html")
    if page is None:
        page=get("https://adastra.autosome.org/soos/","downloads_alternative.html")
    if page:
        from urllib.parse import urljoin
        links=re.findall(r'href=[\"\']([^\"\']+)[\"\']',page.text)
        metadata=[urljoin(page.url,u) for u in links if "Mabel_GTRD_exps" in u]
        for i,url in enumerate(dict.fromkeys(metadata)):
            get(url,f"GTRD_experiments_{i}.tsv")
    get("https://adastra.autosome.org/api/v6/swagger.json","swagger.json")
    get("https://raw.githubusercontent.com/autosome-ru/ADASTRA-pipeline/master/README.md","pipeline_readme.md")
    get("https://www.encodeproject.org/experiments/ENCSR000DKV/?format=json","ENCSR000DKV.json")
    get("https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA324773&result=read_run&fields=run_accession,experiment_accession,sample_accession,fastq_bytes,fastq_md5,fastq_ftp&format=tsv","GSE81945_runs_candidate.tsv")
    print(json.dumps(attempts,indent=2))


if __name__=="__main__":main()
