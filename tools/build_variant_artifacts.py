"""Verify immutable study inputs and generate a bounded feasibility report."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import argparse
from collections import Counter
from src.variant_audit import audit_study

import pandas as pd

from src.utils import sha256_file,write_json

ROOT=Path(__file__).resolve().parents[1]


def build(study=ROOT/"results/variant_study",pilot=ROOT/"results/variant_pilot",report=ROOT/"docs/MINTS_v2_feasibility.md"):
    study,pilot,report=Path(study).resolve(),Path(pilot).resolve(),Path(report).resolve()
    if any(not path.is_relative_to(ROOT) for path in (study,pilot,report)):
        raise ValueError("Study, pilot and report outputs must stay within the repository")
    audit=audit_study(ROOT,study,pilot,check_report=False)
    summary=json.loads((pilot/"summary.json").read_text())
    gate=json.loads((pilot/"gate.json").read_text())
    eligibility=pd.read_csv(pilot/"eligibility.csv")
    inventory=pd.read_csv(study/"dataset_inventory.csv")
    counts=eligibility.groupby("reason").size().sort_index()
    counts.rename("rows").to_csv(study/"exclusion_counts.csv")
    if summary["retained"]!=0 or gate["status"]!="stop":
        raise ValueError("This report describes a stopped empty pilot; a nonempty study needs its own report")
    traces=json.loads((pilot/"eligibility_diagnostics.json").read_text())
    rejection_counts=Counter()
    for trace in traces:rejection_counts.update(trace["candidate_rejections"])
    candidates=sum(t["candidate_positions_checked"] for t in traces)
    rejection_table=pd.DataFrame([dict(first_rejection=k,candidates=v) for k,v in sorted(rejection_counts.items())])
    rejection_table.to_csv(study/"sham_constraint_counts.csv",index=False)
    boundaries=[dict(source_row=t["source_row"],reference_tokens=t["reference_token_count"],
        alternate_tokens=t["alternate_token_count"],first_boundary_difference=t["first_boundary_difference"],
        reference_variant_tokens=json.dumps(t["reference_variant_tokens"]),
        alternate_variant_tokens=json.dumps(t["alternate_variant_tokens"])) for t in traces if "first_boundary_difference" in t]
    pd.DataFrame(boundaries).to_csv(study/"token_boundary_diagnostics.csv",index=False)
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
    report.parent.mkdir(parents=True,exist_ok=True)
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
