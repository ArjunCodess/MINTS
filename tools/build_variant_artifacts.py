"""Verify immutable study inputs and generate a bounded feasibility report."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import argparse
import os
from collections import Counter
from src.variant_audit import audit_study

import pandas as pd

from src.utils import sha256_file,write_json

ROOT=Path(__file__).resolve().parents[1]


def validate_report_target(root,study,pilot,report):
    """Keep presentation output away from frozen and scientific evidence."""
    root,study,pilot,report=(Path(p).resolve() for p in (root,study,pilot,report))
    if any(not path.is_relative_to(root) for path in (study,pilot,report)):
        raise ValueError("Study, pilot and report outputs must stay within the repository")
    if report.suffix.lower()!=".md" or any(report.is_relative_to(p) for p in (study,pilot)):
        raise ValueError("Report must be a Markdown document outside study and pilot evidence")
    if report.exists() and not report.is_file():
        raise ValueError("Report output must be a file")
    baseline=json.loads((study/"baseline_manifest.json").read_text(encoding="utf-8-sig"))
    if report.relative_to(root) in {Path(name) for name in baseline["files"]} or report==root/"docs/MINTS_v2_protocol.md":
        raise ValueError("Report cannot overwrite a frozen baseline or protocol document")


def powershell_path(path):
    return "'"+Path(path).relative_to(ROOT).as_posix().replace("'","''")+"'"


def build(study=ROOT/"results/variant_study",pilot=ROOT/"results/variant_pilot",report=ROOT/"docs/MINTS_v2_feasibility.md"):
    study,pilot,report=Path(study).resolve(),Path(pilot).resolve(),Path(report).resolve()
    if any(not path.is_relative_to(ROOT) for path in (study,pilot,report)):
        raise ValueError("Study, pilot and report outputs must stay within the repository")
    audit=audit_study(ROOT,study,pilot,check_report=False)
    validate_report_target(ROOT,study,pilot,report)
    summary=json.loads((pilot/"summary.json").read_text())
    gate=json.loads((pilot/"gate.json").read_text())
    if summary["retained"]!=0 or gate["status"]!="stop":
        raise ValueError("This report describes a stopped empty pilot; a nonempty study needs its own report")
    eligibility=pd.read_csv(pilot/"eligibility.csv")
    inventory=pd.read_csv(study/"dataset_inventory.csv")
    counts=eligibility.groupby("reason").size().sort_index()
    traces=json.loads((pilot/"eligibility_diagnostics.json").read_text())
    rejection_counts=Counter()
    for trace in traces:rejection_counts.update(trace["candidate_rejections"])
    candidates=sum(t["candidate_positions_checked"] for t in traces)
    def link(name):
        return Path(os.path.relpath(ROOT/"docs"/name,report.parent)).as_posix()
    rejection_table=pd.DataFrame([dict(first_rejection=k,candidates=v) for k,v in sorted(rejection_counts.items())])
    boundaries=[dict(source_row=t["source_row"],reference_tokens=t["reference_token_count"],
        alternate_tokens=t["alternate_token_count"],first_boundary_difference=t["first_boundary_difference"],
        reference_variant_tokens=json.dumps(t["reference_variant_tokens"]),
        alternate_variant_tokens=json.dumps(t["alternate_variant_tokens"])) for t in traces if "first_boundary_difference" in t]
    equal_counts=sum(r["reference_tokens"]==r["alternate_tokens"] for r in boundaries)
    constraint_lines="\n".join(f"| {r.first_rejection} | {r.candidates} |" for r in rejection_table.itertuples())
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

The {len(boundaries)} boundary mismatches concern complete reference/alternate BPE offsets.
In {equal_counts} of these cases, token counts agree even though nucleotide boundaries
move, so matching tensor shape alone would not establish intervention alignment.
The {sum(t["candidate_positions_checked"]>0 for t in traces)} remaining singleton loci had no control meeting the frozen substitution,
context, geometry and motif-preservation rules. Neither matching tolerance nor
query policy was relaxed after this result. The engineered hook fixture passed,
but it does not demonstrate sensitivity of DNABERT-2 on natural variants.

## Why substitution matching failed

The deterministic search checked {candidates} candidate positions. Each row below
counts the first violated constraint in the frozen check order, not independent
failure rates or evidence that relaxing one rule would retain a valid control.

| First candidate rejection | Positions |
| --- | ---: |
{constraint_lines}

These diagnostics use sequences and tokenizer offsets only. They add no model
outcomes, alternative endpoint, relaxed tolerance or confirmation cohort.

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

This report was built from saved evidence with:

```powershell
python tools/build_variant_artifacts.py --study {powershell_path(study)} --pilot {powershell_path(pilot)} --report {powershell_path(report)}
python tools/audit_variant_study.py --study {powershell_path(study)} --pilot {powershell_path(pilot)}
```

Use fresh output directories for every scientific attempt, following the
[reproduction instructions]({link('variant_verification.md')}#fresh-output-reproduction).
The report builder verifies the supplied study, scientific receipts,
historical attempt, metadata hashes and frozen Git baseline. The initial attempt
is retained with source snapshots; its reporting label was corrected to
distinguish unrun controls from failed controls without changing eligibility.

Continuing requires a new exploratory protocol for nucleotide correspondence
and control feasibility, or a larger cohort supporting the frozen rules.
Any revision must be saved before model scoring and must not reinterpret this
empty retained population as a positive result. The [protocol]({link('MINTS_v2_protocol.md')})
and [data inventory]({link('variant_data_feasibility.md')}) describe the required evidence.
"""
    report.parent.mkdir(parents=True,exist_ok=True)
    counts.rename("rows").to_csv(study/"exclusion_counts.csv",lineterminator="\n")
    rejection_table.to_csv(study/"sham_constraint_counts.csv",index=False,lineterminator="\n")
    pd.DataFrame(boundaries).to_csv(study/"token_boundary_diagnostics.csv",index=False,lineterminator="\n")
    report.write_bytes(text.encode())
    outputs={report.relative_to(ROOT).as_posix():sha256_file(report)}
    for name in ("exclusion_counts.csv","sham_constraint_counts.csv","token_boundary_diagnostics.csv"):
        outputs[(study/name).relative_to(ROOT).as_posix()]=sha256_file(study/name)
    write_json(study/"report_manifest.json",dict(generator_sha256=sha256_file(Path(__file__)),
        auditor_sha256=sha256_file(ROOT/"src/variant_audit.py"),
        audit_cli_sha256=sha256_file(ROOT/"tools/audit_variant_study.py"),
        study_inputs={name:sha256_file(study/name) for name in ("dataset_inventory.csv","inspection_ledger.json")},
        scientific_execution_sha256=sha256_file(pilot/"execution.json"),
        baseline_manifest_sha256=sha256_file(study/"baseline_manifest.json"),
        source_retrieval_sha256=sha256_file(study/"source_retrieval.json"),
        protocol_document_sha256=sha256_file(ROOT/"docs/MINTS_v2_protocol.md"),outputs=outputs))
    print("verified baseline, metadata, attempted and final feasibility runs; generated report")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=ROOT/"results/variant_study")
    parser.add_argument("--pilot",type=Path,default=ROOT/"results/variant_pilot")
    parser.add_argument("--report",type=Path,default=ROOT/"docs/MINTS_v2_feasibility.md")
    args=parser.parse_args();build(args.study,args.pilot,args.report)
