"""Freeze the merged baseline and inspect source metadata without variant scores."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd
import requests

from src.utils import sha256_file, write_json

ROOT=Path(__file__).resolve().parents[1]


def prepare(output, baseline):
    output=Path(output).resolve()
    if (output/"baseline_manifest.json").exists():
        raise FileExistsError("Use a fresh study directory")
    output.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(["git","rev-parse",baseline+"^{commit}"],cwd=ROOT,text=True).strip()
    files={}
    for line in subprocess.check_output(["git","ls-tree","-r",commit],cwd=ROOT,text=True).splitlines():
        meta,name=line.split("\t",1)
        files[name]=dict(git_blob=meta.split()[2])
    for name in ("paper/main.pdf","paper/main.tex","results/native_followup/execution.json",
                 "results/native_followup/protocol.json","results/native_endpoint/protocol.json","src/config.py"):
        files[name]["sha256"]=hashlib.sha256(subprocess.check_output(["git","show",commit+":"+name],cwd=ROOT)).hexdigest()
    write_json(output/"baseline_manifest.json",dict(commit=commit,tag=baseline,files=files,
        scope="exact Git tree retains manuscript, configs, protocols, results and execution receipts",
        scientific_stop="original masked-flank sensitivity gate remains failed",
        generator_sha256=sha256_file(Path(__file__))))
    sources=[]
    def retrieve(name,url,params=None):
        entry=dict(name=name,url=url,requested_parameters=params)
        try:
            response=requests.get(url,params=params,timeout=20)
            entry.update(status=response.status_code,resolved_url=response.url)
            response.raise_for_status()
            path=output/(name+".json" if "json" in response.headers.get("Content-Type","") or name in ("adastra_tf_metadata","zenodo_release") else name+".txt")
            path.write_bytes(response.content)
            entry.update(path=path.name,sha256=sha256_file(path))
            return response
        except requests.RequestException as exc:
            entry["error"]=str(exc)
            return None
        finally:
            sources.append(entry)
    tf=retrieve("adastra_tf_metadata","https://adastra.autosome.org/api/v6/browse/tf",dict(size=1200))
    retrieve("adastra_readme","https://adastra.autosome.org/assets/readme/readme.mabel.txt")
    retrieve("zenodo_release","https://zenodo.org/api/records/14174114")
    retrieve("alleleDB_access","https://alleledb.gersteinlab.org/download/")
    rows=[]
    if tf is not None:
        matches=[r for r in tf.json().get("results",[]) if r.get("name")=="CTCF_HUMAN"]
        if len(matches)!=1:
            raise ValueError("ADASTRA CTCF metadata missing or ambiguous")
        ctcf=matches[0]
        rows.append(dict(dataset="ADASTRA Mabel v6.1",genome_build="hg38",candidate_variants=ctcf["aggregated_snps_count"],
            significant_005=ctcf["aggregated_snps_count005"],experiments=ctcf["experiments_count"],eligible_loci=None,
            usable_donors=None,confirmation_status="unresolved: no per-locus QC, independent donor design or power verified",
            source="https://adastra.autosome.org/api/v6/browse/tf"))
    rows.extend([
        dict(dataset="GSE81945 Table S3",genome_build="hg19",candidate_variants=16,significant_005=None,experiments=2,
            eligible_loci=11,usable_donors=1,confirmation_status="exploratory only; 11 singleton heterozygous loci before token/control exclusions",
            source="https://ars.els-cdn.com/content/image/1-s2.0-S221112471631628X-mmc1.pdf"),
        dict(dataset="AlleleDB",genome_build="hg19",candidate_variants=None,significant_005=None,experiments=None,
            eligible_loci=None,usable_donors=None,confirmation_status="access failed; do not equate published all-TF totals with CTCF eligibility",
            source="https://www.nature.com/articles/ncomms11101"),
        dict(dataset="BaalChIP ENCODE panel",genome_build="unverified for proposed import",candidate_variants=None,
            significant_005=None,experiments=None,eligible_loci=None,usable_donors=None,
            confirmation_status="potential source; 548 all-TF samples/14 cell lines are not independent CTCF variant counts",
            source="https://doi.org/10.1186/s13059-017-1165-7")])
    pd.DataFrame(rows).to_csv(output/"dataset_inventory.csv",index=False)
    write_json(output/"source_retrieval.json",dict(checked_at=datetime.now(timezone.utc).isoformat(),sources=sources,
        outcome_inspection="aggregate metadata only; no ADASTRA variant outcomes or sequences retrieved",
        conclusion="larger candidate resource accessible; adequately powered independent CTCF confirmation not yet established",
        code_sha256=sha256_file(Path(__file__))))
    ledger=dict(baseline_commit=commit,entries=[
        dict(dataset="existing hg38 CTCF audit",membership="all existing genomic assay manifests",role="previously inspected; never fresh by default"),
        dict(dataset="native pilot",chromosomes=["chr18","chr19"],role="inspected discovery"),
        dict(dataset="native calibration",chromosomes=["chr14","chr15","chr16","chr17"],role="inspected frequency/recovery"),
        dict(dataset="prior audit confirmation candidates",chromosomes=["chr20","chr21"],role="already inspected"),
        dict(dataset="GSE81945",membership_sha256=sha256_file(ROOT/"results/native_followup/ctcf_allele_effects.csv"),role="published effects inspected; feasibility only"),
        dict(dataset="ADASTRA",role="metadata inspected only; confirmation membership unassigned; source overlap unverified")],
        confirmation_membership=None,freshness_verified=False,
        separation_requirements=["coordinate overlap within same genome build","exact/reverse-complement sequence identity",
            "homology review","shared donor/biosample and source experiments","publication/source overlap"],
        pretraining_overlap="not excluded; fresh experimental outcomes do not imply unseen reference sequence")
    write_json(output/"inspection_ledger.json",ledger)
    print(json.dumps(dict(baseline=commit,inventory_rows=len(rows),confirmation_ready=False),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/variant_study")
    parser.add_argument("--baseline",default="mints-audit-v1-merged-2026-10-05")
    args=parser.parse_args();prepare(args.output,args.baseline)
