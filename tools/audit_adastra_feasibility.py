"""Verify downloaded-cohort evidence offline, without genomic or model caches."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
from collections import Counter
from dataclasses import asdict

import pandas as pd

from src.adastra_candidates import select_candidates
from src.utils import sha256_file
from src.variant_audit import within
from src.variant_protocol import VariantProtocol
from tools.download_adastra import MD5,SIZE,URL

ROOT=Path(__file__).resolve().parents[1]


def audit(output,root=ROOT):
    output=within(root,output)
    read=lambda name:json.loads((output/name).read_text(encoding="utf-8-sig"))
    history=root/"results/adastra_import_attempt"
    if history.exists():
        failed=json.loads((history/"execution.json").read_text())
        if failed["status"]!="failed" or failed["source_changed_during_run"] is not False:
            raise ValueError("Import attempt history changed")
        for name,value in failed["source_sha256"].items():
            if sha256_file(within(history/"source_snapshot",name))!=value:raise ValueError("Import source snapshot changed")
        for name,value in failed["artifacts"].items():
            if sha256_file(within(history,name))!=value:raise ValueError("Import attempt artifact changed")
    execution=read("execution.json");protocol=read("protocol.json");source=read("source.json")
    if execution["status"]!="completed" or execution["source_changed_during_run"] is not False:
        raise ValueError("Incomplete or changed cohort screen")
    required={"protocol.json","source.json","candidates.csv.gz","membership.csv","eligibility.csv",
              "diagnostics.json","sequence_cases.json","summary.json","inspection_ledger.json"}
    if not required<=set(execution["artifacts"]):raise ValueError("Missing cohort evidence")
    for name,value in execution["artifacts"].items():
        if sha256_file(within(output,name))!=value:raise ValueError("Cohort evidence changed: "+name)
    if execution["protocol_sha256"]!=sha256_file(output/"protocol.json") or protocol["source_sha256"]!=execution["source_sha256"]:
        raise ValueError("Frozen cohort source sets differ")
    if protocol["sequence_constraints"]!=asdict(VariantProtocol()) or protocol["selection_seed"]!=1731 or protocol["window_width_bp"]!=204:
        raise ValueError("Frozen sequence constraints or membership policy changed")
    if not {"src/adastra_candidates.py","tools/run_adastra_feasibility.py","tools/download_adastra.py","tools/audit_adastra_feasibility.py"}<=set(execution["source_sha256"]):
        raise ValueError("Missing cohort implementation hashes")
    for name,value in execution["source_sha256"].items():
        if sha256_file(within(root,name))!=value:raise ValueError("Cohort implementation changed: "+name)
    download=read("download.json")
    if (download["status"]!="completed" or download["publisher_md5"]!=MD5 or download["bytes"]!=SIZE
            or download["url"]!=URL or download["sha256"]!=source["archive_sha256"]):
        raise ValueError("Download receipt differs from pinned source")
    candidates=pd.read_csv(output/"candidates.csv.gz");membership=pd.read_csv(output/"membership.csv")
    if candidates.variant_id.duplicated().any() or list(candidates.source_row)!=list(range(1,len(candidates)+1)):
        raise ValueError("Full candidate denominator changed")
    expected=select_candidates(candidates,protocol["selection_limit"],protocol["selection_seed"])
    pd.testing.assert_frame_equal(membership,expected,check_dtype=False)
    eligibility=pd.read_csv(output/"eligibility.csv")
    pd.testing.assert_frame_equal(eligibility[list(membership.columns)],membership,check_dtype=False)
    for column in ("reference_verified","sequence_control_feasible"):
        if not eligibility[column].map(lambda v:type(v) is bool).all():raise ValueError("Nonboolean eligibility flag")
    diagnostics=read("diagnostics.json");cases=read("sequence_cases.json");summary=read("summary.json")
    if [t["variant_id"] for t in diagnostics]!=list(membership.variant_id):raise ValueError("Diagnostic membership differs")
    for trace,row in zip(diagnostics,eligibility.to_dict("records")):
        checked=trace["candidate_positions_checked"];rejected=trace["candidate_rejections"]
        if type(checked) is not int or checked<0 or any(type(n) is not int or n<0 for n in rejected.values()):
            raise ValueError("Invalid sham candidate accounting")
        if checked!=sum(rejected.values())+int(row["sequence_control_feasible"]) or trace["reason"]!=row["reason"]:
            raise ValueError("Sham accounting or reason differs")
    retained=eligibility[eligibility.sequence_control_feasible]
    if [c["variant_id"] for c in cases]!=list(retained.variant_id):raise ValueError("Sequence case membership differs")
    if (summary["source_records"]!=len(candidates) or source["rows"]!=len(candidates)
            or summary["screened"]!=len(membership) or summary["sequence_controls"]!=len(cases)
            or summary["reference_verified"]!=int(eligibility.reference_verified.sum())
            or summary["reasons"]!=dict(Counter(eligibility.reason))):raise ValueError("Cohort population accounting differs")
    if (source["outcomes_used"] is not False or summary["confirmation_ready"] is not False
            or summary["native_inference"]!="not run by design" or summary["head_search"]!="not run"
            or summary["biological_eligible_loci"] is not None or summary["usable_donors"] is not None
            or any(c["biological_confirmation_eligible"] is not False for c in cases)):
        raise ValueError("Engineering screen cannot establish biological inference or confirmation")
    ledger=read("inspection_ledger.json")
    if (ledger["fresh_confirmation_membership"] is not None or ledger["outcomes_used"] is not False
            or ledger["source_records"]!=len(candidates)
            or ledger["selected_membership_sha256"]!=sha256_file(output/"membership.csv")):
        raise ValueError("Inspection ledger disagrees")
    if (output/"report_manifest.json").exists():
        report=read("report_manifest.json")
        for name,path in {"generator_sha256":root/"tools/build_adastra_report.py",
                          "auditor_sha256":root/"tools/audit_adastra_feasibility.py",
                          "execution_sha256":output/"execution.json",
                          "report_sha256":within(root,report["report_path"])}.items():
            if report[name]!=sha256_file(path):raise ValueError("Cohort report changed: "+name)
    return dict(status="verified",source_records=len(candidates),screened=len(membership),sequence_controls=len(cases),confirmation_ready=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/adastra_exploratory")
    args=parser.parse_args()
    try:print(json.dumps(audit(args.output),indent=2))
    except (ValueError,KeyError,OSError,TypeError,AssertionError) as exc:
        print(json.dumps(dict(status="failed",error=str(exc)),indent=2));sys.exit(1)
