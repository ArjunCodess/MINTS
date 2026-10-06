"""Try the official rsID search for missing SNP details; validate GEO membership."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import re
import requests
import pandas as pd
from src.utils import write_json,sha256_file

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/"results/variant_provenance"
    cache=ROOT/"data/adastra/provenance_v1"
    attempts=json.loads((output/"snp_downloads.json").read_text())
    missing=[r for r in attempts if r["status"]=="failed"]
    def get(r):
        rs=r["url"].split("/")[-2]
        url=f"https://adastra.autosome.org/api/v6/search/snps/rs/{rs}"
        record=dict(variant_id=r["variant_id"],url=url,access_utc=pd.Timestamp.now(tz="UTC").isoformat())
        try:
            response=requests.get(url,timeout=(15,35));record["http_status"]=response.status_code
            response.raise_for_status();value=response.json()
            path=cache/f"search_{rs}.json";path.write_bytes(response.content)
            record.update(status="downloaded",bytes=len(response.content),sha256=sha256_file(path),
                response=value,interpretation="search availability only; no read counts inferred")
        except Exception as exc:record.update(status="failed",reason=f"{type(exc).__name__}: {exc}")
        return record
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(get,missing))
    write_json(output/"alternative_requests.json",records)
    text=gzip.decompress((cache/"GSE81945.soft.gz").read_bytes()).decode()
    samples=re.findall(r'^\^SAMPLE = (GSM\d+)\s*$',text,re.M)
    ena=pd.read_csv(cache/"PRJNA323481_runs.tsv",sep="\t")
    write_json(output/"geo_source_validation.json",dict(series="GSE81945",samples=samples,
        project="PRJNA323481",study="SRP075768",fastq_bytes=int(ena.fastq_bytes.astype(str).str.split(";").apply(lambda x:sum(map(int,x))).sum()),
        runs=ena.run_accession.tolist(),source_sha256=sha256_file(cache/"GSE81945.soft.gz"),
        correction="geo_relations.json used broad regex that included a truncated GSM fragment; only exact SAMPLE lines identify samples",
        excluded_candidate="PRJNA324773 initial query was not linked by this GEO deposit; preserved but not used"))
    print(json.dumps(dict(alternative_requests=len(records),statuses=[(r["http_status"],r.get("response",{}).get("total")) for r in records]),indent=2))


if __name__=="__main__":main()
