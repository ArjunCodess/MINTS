"""Offline byte and semantic verification of the bounded variant study."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

from .variant_protocol import VariantProtocol, feasibility_gate, confirmation_readiness
from .variant_quality import biological_qc


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def within(root,name):
    path=(Path(root)/name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Evidence path escapes its root: "+str(name))
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_execution(root,folder,historical=False):
    root,folder=Path(root),Path(folder)
    receipt=read(folder/"execution.json")
    if receipt.get("status")!="completed" or receipt.get("source_changed_during_run") is not False:
        raise ValueError("Scientific execution incomplete or changed during run")
    if not receipt.get("source_sha256") or not receipt.get("artifacts"):
        raise ValueError("Scientific source and output hashes are required")
    scientific_sources={"src/variant_assay.py","src/variant_protocol.py","src/variant_statistics.py","tools/run_variant_pilot.py",
        "src/native_endpoint.py","src/native_followup.py","src/assay_alignment.py","src/assay_stats.py",
        "src/modeling.py","src/config.py","src/motif_scoring.py","src/controlled_edits.py","src/utils.py","src/variant_quality.py"}
    if not scientific_sources<=set(receipt["source_sha256"]):
        raise ValueError("Required scientific source hashes missing")
    for name,expected in receipt["source_sha256"].items():
        source=within(root,name)
        saved=within(folder/"source_snapshot",name)
        if historical and saved.exists():source=saved
        if digest(source)!=expected:raise ValueError("Scientific source changed: "+name)
    for name,expected in receipt["artifacts"].items():
        if digest(within(folder,name))!=expected:raise ValueError("Scientific output changed: "+name)
    if receipt["protocol_sha256"]!=digest(folder/"protocol.json"):
        raise ValueError("Protocol receipt hash differs")
    protocol=read(folder/"protocol.json")
    if protocol["source_sha256"]!=receipt["source_sha256"]:
        raise ValueError("Protocol and execution source sets differ")
    return receipt


def audit_pilot(root,pilot):
    root,pilot=Path(root),Path(pilot)
    receipt=verify_execution(root,pilot)
    required={"protocol.json","eligibility.csv","eligibility_diagnostics.json","membership.json",
              "biological_qc.json","checkpoint.json","engineered_control.json","scores.csv",
              "query_diagnostics.json","summary.json","gate.json","power.json","confirmation_readiness.json"}
    if not required<=set(receipt["artifacts"]):
        raise ValueError("Required scientific artifacts missing from receipt")
    for attempt in sorted((pilot/"attempts").glob("*")):
        if attempt.is_dir():verify_execution(root,attempt,historical=True)
    frozen=read(pilot/"protocol.json")
    protocol=VariantProtocol(**{f.name:frozen[f.name] for f in fields(VariantProtocol)})
    if any(frozen.get(k)!=v for k,v in protocol.record().items()):
        raise ValueError("Frozen endpoint/control rules disagree with protocol implementation")
    source=root/"results/native_followup/ctcf_allele_effects.csv"
    manifest=root/"results/native_followup/allele_manifest.json"
    if digest(source)!=frozen["input_sha256"] or digest(manifest)!=frozen["biological_source_manifest_sha256"]:
        raise ValueError("Biological input hashes disagree with protocol")
    if read(manifest)["table_sha256"]!=digest(source):
        raise ValueError("Biological extraction receipt disagrees")
    table=pd.read_csv(source);eligibility=pd.read_csv(pilot/"eligibility.csv")
    if eligibility.source_row.duplicated().any() or list(eligibility.source_row)!=list(table.source_row):
        raise ValueError("Eligibility must account for every source row exactly once and in order")
    if not eligibility.retained.map(lambda v:isinstance(v,bool)).all():
        raise ValueError("Eligibility retained flags must be boolean")
    if list(eligibility.locus_group)!=list(table.locus_group):
        raise ValueError("Eligibility locus groups changed")
    traces=read(pilot/"eligibility_diagnostics.json")
    if [r["source_row"] for r in traces]!=list(table.source_row):
        raise ValueError("Diagnostic population differs from biological source")
    for trace,row in zip(traces,eligibility.to_dict("records")):
        if trace["reason"]!=row["reason"]:
            raise ValueError("Diagnostic and eligibility reasons differ")
        checked=trace["candidate_positions_checked"]
        rejected=sum(trace["candidate_rejections"].values())
        if type(checked) is not int or checked<0 or any(type(n) is not int or n<0 for n in trace["candidate_rejections"].values()):
            raise ValueError("Invalid candidate accounting")
        if rejected+int(row["retained"])!=checked:
            raise ValueError("Every checked sham candidate needs one first rejection or acceptance")
    expected_qc=[dict(source_row=r["source_row"],**biological_qc(r)) for r in table.to_dict("records")]
    if read(pilot/"biological_qc.json")!=expected_qc:
        raise ValueError("Biological QC does not match available source evidence")
    summary=read(pilot/"summary.json");membership=read(pilot/"membership.json")
    count=int(eligibility.retained.sum())
    sizes=table.groupby("locus_group").size()
    singleton=int(sum(r["zygosity"]=="heterozygous" and sizes[r["locus_group"]]==1 for r in table.to_dict("records")))
    if summary["scanned"]!=len(table) or summary["singleton_heterozygous_loci"]!=singleton or summary["retained"]!=count:
        raise ValueError("Summary population counts disagree with source and eligibility")
    if summary["retention"]!=(count/singleton if singleton else 0) or len(membership["cases"])!=count:
        raise ValueError("Retention or membership denominator differs")
    if membership["biological_effects_used_for_selection"] is not False:
        raise ValueError("Biological outcomes cannot select sequence eligibility")
    scores=pd.read_csv(pilot/"scores.csv")
    if len(scores)!=count:raise ValueError("Score population differs from retained membership")
    gate=read(pilot/"gate.json")
    if gate!=feasibility_gate(summary,protocol):raise ValueError("Saved gate differs from frozen rules")
    if count==0:
        if (summary["controls_passed"] is not None or summary["clusters"]!=0
                or any(summary.get(k) is not None for k in ("mean","ci_low","ci_high","binding_rho","binding_ci_low","binding_ci_high"))
                or read(pilot/"query_diagnostics.json") or read(pilot/"checkpoint.json")["status"]!="not loaded; no sequence-eligible matched cases"):
            raise ValueError("Empty population must not claim native measurements")
    if gate["status"]=="stop":
        if summary["selected_head"] is not None or read(pilot/"power.json")["status"]!="not estimable":
            raise ValueError("Stopped pilot cannot select heads or estimate intervention power")
        if any((pilot/n).exists() for n in ("selected_head.json","discovery_heads.json")):
            raise ValueError("Stopped pilot contains head-search outputs")
    readiness=confirmation_readiness(dict(pilot_passed=gate["status"]=="eligible_for_discovery"))
    if read(pilot/"confirmation_readiness.json")!=readiness:
        raise ValueError("Confirmation readiness disagrees with verified study state")
    return dict(scanned=len(table),retained=count,decision=gate["status"],historical_attempts=len(list((pilot/"attempts").glob("*/execution.json"))))


def audit_inventory(study):
    study=Path(study)
    inventory=pd.read_csv(study/"dataset_inventory.csv")
    tf=[r for r in read(study/"adastra_tf_metadata.json")["results"] if r["name"]=="CTCF_HUMAN"]
    ctcf=inventory[inventory.dataset=="ADASTRA Mabel v6.1"]
    if len(tf)!=1 or len(ctcf)!=1:raise ValueError("CTCF metadata missing or duplicated")
    for column,key in (("candidate_variants","aggregated_snps_count"),("significant_005","aggregated_snps_count005"),("experiments","experiments_count")):
        if ctcf.iloc[0][column]!=tf[0][key]:raise ValueError("Inventory differs from retrieved CTCF counts")
    if ctcf.iloc[0].genome_build!="hg38":raise ValueError("ADASTRA inventory build differs from the retrieved release")
    if pd.notna(ctcf.iloc[0].eligible_loci) or pd.notna(ctcf.iloc[0].usable_donors):
        raise ValueError("Aggregate metadata cannot establish eligible independent loci or donors")


def audit_study(root,study=None,pilot=None,check_report=True):
    root=Path(root).resolve()
    study=within(root,study or "results/variant_study")
    pilot=within(root,pilot or "results/variant_pilot")
    result=audit_pilot(root,pilot)
    baseline=read(study/"baseline_manifest.json")
    tags=subprocess.check_output(["git","tag","--list",baseline["tag"]],cwd=root,text=True).strip()
    reference=baseline["tag"] if tags else baseline["commit"]
    commit=subprocess.check_output(["git","rev-parse",reference+"^{commit}"],cwd=root,text=True).strip()
    if commit!=baseline["commit"]:raise ValueError("Baseline tag moved")
    tree={line.split("\t",1)[1]:line.split("\t",1)[0].split()[2]
          for line in subprocess.check_output(["git","ls-tree","-r",commit],cwd=root,text=True).splitlines()}
    if tree!={name:r["git_blob"] for name,r in baseline["files"].items()}:
        raise ValueError("Frozen baseline tree differs")
    for name,record in baseline["files"].items():
        if "sha256" in record:
            blob=subprocess.check_output(["git","show",commit+":"+name],cwd=root)
            if hashlib.sha256(blob).hexdigest()!=record["sha256"]:raise ValueError("Baseline file digest differs: "+name)
    retrieval=read(study/"source_retrieval.json")
    if retrieval["code_sha256"]!=digest(root/"tools/prepare_variant_study.py"):
        raise ValueError("Metadata retrieval source changed")
    for source in retrieval["sources"]:
        if "path" in source and digest(within(study,source["path"]))!=source["sha256"]:
            raise ValueError("Retrieved metadata changed")
    audit_inventory(study)
    ledger=read(study/"inspection_ledger.json")
    if ledger["baseline_commit"]!=commit or ledger["confirmation_membership"] is not None or ledger["freshness_verified"] is not False:
        raise ValueError("Inspection ledger claims unverified confirmation freshness")
    if check_report:
        report=read(study/"report_manifest.json")
        expected={"scientific_execution_sha256":pilot/"execution.json","baseline_manifest_sha256":study/"baseline_manifest.json",
                  "source_retrieval_sha256":study/"source_retrieval.json","protocol_document_sha256":root/"docs/MINTS_v2_protocol.md",
                  "generator_sha256":root/"tools/build_variant_artifacts.py"}
        for name,path in expected.items():
            if report[name]!=digest(path):raise ValueError("Report input/source hash differs: "+name)
        if report.get("auditor_sha256")!=digest(root/"src/variant_audit.py"):
            raise ValueError("Report auditor source changed")
        if report.get("audit_cli_sha256")!=digest(root/"tools/audit_variant_study.py"):
            raise ValueError("Audit command source changed")
        required_outputs={(study/name).relative_to(root).as_posix() for name in
            ("exclusion_counts.csv","sham_constraint_counts.csv","token_boundary_diagnostics.csv")}
        if not required_outputs<=set(report["outputs"]) or len(report["outputs"])!=4 or sum(name.endswith(".md") for name in report["outputs"])!=1:
            raise ValueError("Report must bind its document and all three diagnostic tables")
        for name,expected_hash in report["outputs"].items():
            if digest(within(root,name))!=expected_hash:raise ValueError("Generated report changed: "+name)
        study_inputs=report.get("study_inputs",{})
        if not {"dataset_inventory.csv","inspection_ledger.json"}<=set(study_inputs):raise ValueError("Report must bind the inventory and inspection ledger")
        for name,expected_hash in study_inputs.items():
            if digest(within(study,name))!=expected_hash:raise ValueError("Study input changed: "+name)
    return dict(status="verified",**result,baseline_commit=commit,confirmation_ready=False,
                scope="offline byte and semantic audit; no model loading, network, new cohort or relaxed controls")
