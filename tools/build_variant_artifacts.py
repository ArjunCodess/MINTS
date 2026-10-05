"""Verify immutable study inputs and generate a bounded feasibility report."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import subprocess

import pandas as pd

from src.utils import sha256_file,write_json

ROOT=Path(__file__).resolve().parents[1]


def verify_run(folder):
    receipt=json.loads((folder/"execution.json").read_text())
    if receipt["status"]!="completed" or receipt["source_changed_during_run"]:
        raise ValueError("Scientific execution incomplete or changed during run")
    for name,digest in receipt["source_sha256"].items():
        if sha256_file(ROOT/name)!=digest:
            raise ValueError("Scientific source changed: "+name)
    for name,digest in receipt["artifacts"].items():
        if sha256_file(folder/name)!=digest:
            raise ValueError("Scientific output changed: "+name)
    return receipt


def build():
    pilot,study=ROOT/"results/variant_pilot",ROOT/"results/variant_study"
    receipt=verify_run(pilot)
    prior=pilot/"attempts/initial"
    if prior.exists():
        initial=json.loads((prior/"execution.json").read_text())
        for name,digest in initial["source_sha256"].items():
            source=prior/"source_snapshot"/name
            if not source.exists():source=ROOT/name
            if sha256_file(source)!=digest:raise ValueError("Historical attempted-run source changed: "+name)
        for name,digest in initial["artifacts"].items():
            if sha256_file(prior/name)!=digest:raise ValueError("Historical attempted-run output changed: "+name)
    baseline=json.loads((study/"baseline_manifest.json").read_text())
    reference=baseline["tag"] if subprocess.check_output(["git","tag","--list",baseline["tag"]],cwd=ROOT,text=True).strip() else baseline["commit"]
    commit=subprocess.check_output(["git","rev-parse",reference+"^{commit}"],cwd=ROOT,text=True).strip()
    if commit!=baseline["commit"]:raise ValueError("Baseline tag moved")
    tree={line.split("\t",1)[1]:line.split("\t",1)[0].split()[2]
          for line in subprocess.check_output(["git","ls-tree","-r",commit],cwd=ROOT,text=True).splitlines()}
    if tree!={name:r["git_blob"] for name,r in baseline["files"].items()}:
        raise ValueError("Baseline tree differs from frozen manifest")
    retrieval=json.loads((study/"source_retrieval.json").read_text())
    if retrieval["code_sha256"]!=sha256_file(ROOT/"tools/prepare_variant_study.py"):
        raise ValueError("Metadata retrieval source changed")
    for source in retrieval["sources"]:
        if "path" in source and sha256_file(study/source["path"])!=source["sha256"]:
            raise ValueError("Retrieved source metadata changed")
    summary=json.loads((pilot/"summary.json").read_text())
    gate=json.loads((pilot/"gate.json").read_text())
    eligibility=pd.read_csv(pilot/"eligibility.csv")
    inventory=pd.read_csv(study/"dataset_inventory.csv")
    counts=eligibility.groupby("reason").size().sort_index()
    counts.rename("rows").to_csv(study/"exclusion_counts.csv")
    reason_lines="\n".join(f"| {reason} | {n} |" for reason,n in counts.items())
    ctcf=inventory[inventory.dataset=="ADASTRA Mabel v6.1"]
    candidate="unavailable" if ctcf.empty else f"{int(ctcf.iloc[0].candidate_variants):,}"
    text=f"""# MINTS v2 feasibility result

The frozen natural-variant protocol retained {summary['retained']} matched cases
from {summary['singleton_heterozygous_loci']} singleton heterozygous loci. Its
decision is **{gate['status']}**. Native distribution scoring, real-case rescue
controls and head selection were not run because no matched case survived.
This measures control eligibility, not model sensitivity or biological absence.

## Population accounting

All {summary['scanned']} published rows were audited before model inference.

| Exclusion | Rows |
| --- | ---: |
{reason_lines}

The seven boundary mismatches concern complete reference/alternate BPE offsets.
The four remaining singleton loci had no control meeting the frozen substitution,
context, geometry and motif-preservation rules. Neither matching tolerance nor
query policy was relaxed after this result. The engineered hook fixture passed,
but it does not demonstrate sensitivity of DNABERT-2 on natural variants.

## Biological and confirmation limits

ADASTRA's metadata reports {candidate} CTCF candidate records. Its release is
accessible, but eligible independent donors/loci, source overlap and intervention
power remain unverified. Aggregate source counts do not authorize confirmation.
GSE81945 supplies pooled reads; phase, mapping-bias, dosage and replicate checks
remain missing. No native model prediction of binding direction is claimed.

There is no selected head or discovery intervention variance, so a powered
confirmation size cannot be estimated from this run. `power.json` labels its
normal-approximation scenarios illustrative. `confirmation_readiness.json`
keeps confirmation disabled; the old audit's inspected cohorts are excluded
from freshness claims.

## Reproduction and next boundary

Use fresh output directories for every scientific attempt:

```powershell
python tools/prepare_variant_study.py --output results/variant_study_new
python tools/run_variant_pilot.py --output results/variant_pilot_new --device cuda
python tools/build_variant_artifacts.py
```

The report builder verifies the saved default study, scientific receipts,
historical attempt, metadata hashes and frozen Git baseline. The initial attempt
is retained with source snapshots; its reporting label was corrected to
distinguish unrun controls from failed controls without changing eligibility.

Continuing requires a new exploratory protocol for nucleotide correspondence
and control feasibility, or a larger cohort supporting the frozen rules.
Any revision must be saved before model scoring and must not reinterpret this
empty retained population as a positive result. The [protocol](MINTS_v2_protocol.md)
and [data inventory](variant_data_feasibility.md) describe the required evidence.
"""
    report=ROOT/"docs/MINTS_v2_feasibility.md"
    report.write_bytes(text.encode())
    outputs={"docs/MINTS_v2_feasibility.md":sha256_file(report),
             "results/variant_study/exclusion_counts.csv":sha256_file(study/"exclusion_counts.csv")}
    write_json(study/"report_manifest.json",dict(generator_sha256=sha256_file(Path(__file__)),
        scientific_execution_sha256=sha256_file(pilot/"execution.json"),
        baseline_manifest_sha256=sha256_file(study/"baseline_manifest.json"),
        source_retrieval_sha256=sha256_file(study/"source_retrieval.json"),
        protocol_document_sha256=sha256_file(ROOT/"docs/MINTS_v2_protocol.md"),outputs=outputs))
    print("verified baseline, metadata, attempted and final feasibility runs; generated report")


if __name__=="__main__":build()
