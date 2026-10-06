"""Resolve official experiment metadata links and quantify raw-read storage."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gzip
import hashlib
import json
import re
import shutil
import time
import requests
import pandas as pd
from src.utils import write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    cache=ROOT/"data/adastra/provenance_v1"
    receipt=ROOT/"results/variant_provenance"
    records=[]
    def get(url,name):
        path=cache/name
        if path.exists():raise FileExistsError(path)
        item=dict(url=url,name=name,access_utc=pd.Timestamp.now(tz="UTC").isoformat())
        try:
            response=requests.get(url,timeout=(15,40));item["http_status"]=response.status_code
            response.raise_for_status();path.write_bytes(response.content)
            item.update(status="downloaded",bytes=len(response.content),sha256=hashlib.sha256(response.content).hexdigest())
            return response
        except Exception as exc:item.update(status="failed",reason=f"{type(exc).__name__}: {exc}")
        finally:
            records.append(item);write_json(receipt/"extended_downloads.json",records)
    get("https://adastra.autosome.org/assets/exps/ADASTRA_GTRD_exps.bill_cipher.tsv","GTRD_experiments.tsv")
    soft=get("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE81nnn/GSE81945/soft/GSE81945_family.soft.gz","GSE81945.soft.gz")
    control=get("https://www.encodeproject.org/experiments/ENCSR000DKW/?format=json","ENCSR000DKW.json")
    get("https://raw.githubusercontent.com/bmvdgeijn/WASP/master/README.md","WASP_readme.md")
    # Resolve the association between the GEO deposit and ENA, rather than
    # assigning a guessed project accession to the experiment.
    if soft is not None:
        text=gzip.decompress(soft.content).decode()
        relations=sorted(set(re.findall(r'(?:PRJNA|SRP)\d+',text)))
        write_json(receipt/"geo_relations.json",dict(series="GSE81945",accessions=relations,
            samples=sorted(set(re.findall(r'GSM\d+',text))),soft_sha256=hashlib.sha256(soft.content).hexdigest()))
        for accession in relations:
            get(f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={accession}&result=read_run&fields=run_accession,experiment_accession,sample_accession,fastq_bytes,fastq_md5,fastq_ftp&format=tsv",f"{accession}_runs.tsv")
    if (cache/"GTRD_experiments.tsv").exists():
        table=pd.read_csv(cache/"GTRD_experiments.tsv",sep="\t",dtype=str,keep_default_na=False)
        table.head(5).to_csv(receipt/"experiment_schema_example.tsv",sep="\t",index=False,lineterminator="\n")
        write_json(receipt/"experiment_schema.json",dict(rows=len(table),columns=list(table),
            version_warning="live v6.1.1 page labels Mabel but links a bill_cipher filename; overlap/version requires per-variant resolution",
            source_sha256=hashlib.sha256((cache/"GTRD_experiments.tsv").read_bytes()).hexdigest()))
    experiment=json.loads((cache/"ENCSR000DKV.json").read_text())
    control_experiment=json.loads((cache/"ENCSR000DKW.json").read_text()) if control is not None else {}
    def fastq(e):return [dict(accession=f["accession"],bytes=f.get("file_size"),md5=f.get("md5sum"),
        href=f.get("href"),biological_replicates=f.get("biological_replicates")) for f in e.get("files",[]) if f.get("file_format")=="fastq"]
    files=fastq(experiment)+fastq(control_experiment)
    total=sum(f["bytes"] or 0 for f in files)
    reps=[]
    for r in experiment["replicates"]:
        biosample=r.get("library",{}).get("biosample",{})
        reps.append(dict(biological_replicate=r.get("biological_replicate_number"),technical_replicate=r.get("technical_replicate_number"),
            biosample=biosample.get("accession"),donor=biosample.get("donor",{}).get("accession")))
    write_json(receipt/"biological_inventory.json",dict(encode_experiment="ENCSR000DKV",control="ENCSR000DKW",
        role="previously inspected GM12878 assay; not fresh confirmation",replicates=reps,raw_files=files,
        raw_fastq_bytes=total,disk_free_bytes=shutil.disk_usage(ROOT).free,
        conservative_working_space_bytes=30*1024**3,
        working_space_basis="3.2 GB genome plus aligner index, compressed reads, remapped reads and multiple BAM/sort stages; 30 GiB is a planning floor, not measured allocation",
        raw_processing="not started; insufficient spare disk for reference alignment and WASP remapping working set",
        workflow="WASP allele-swap/remap, discard reads with changed mapping; requires verified donor variants, replicate count extraction and explicit CNV/coverage filters",
        missing=["membership-specific experiment links","verified individual genotype and phase","replicate ChIP/input allele counts",
            "mapping-bias-corrected alignments","per-case dosage and uncertainty","independent source separation"],
        statuses=dict(archive="downloaded, MD5/SHA256 verified; imported coordinates and reference windows verified",
            sequence_population="exploratory only; QC eligibility unknown",biological_qc_cases=None,
            independent_donors=None,discovery_eligible="sequence method development only",confirmation_eligible=False)))
    print(json.dumps(json.loads((receipt/"biological_inventory.json").read_text()),indent=2))


if __name__=="__main__":main()
