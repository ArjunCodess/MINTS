"""Reject internally inconsistent study records even when hashes are updated."""
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from src.variant_audit import audit_pilot,audit_inventory,digest,within
from src.variant_assay import js_divergence,validate_variant,prepare_variant
from src.variant_protocol import VariantProtocol,feasibility_gate
from src.variant_quality import separation_audit,sequence_identity
from src.variant_statistics import cluster_summary,simulate_cluster_power

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def copied_pilot(tmp_path):
    root=tmp_path/"copy";pilot=root/"results/variant_pilot"
    shutil.copytree(ROOT/"results/variant_pilot",pilot)
    receipt=json.loads((pilot/"execution.json").read_text())
    for name in receipt["source_sha256"]:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
    biological=root/"results/native_followup";biological.mkdir(parents=True)
    for name in ("ctcf_allele_effects.csv","allele_manifest.json"):
        shutil.copyfile(ROOT/"results/native_followup"/name,biological/name)
    return root,pilot


def edit_artifact(pilot,name,edit):
    path=pilot/name;data=json.loads(path.read_text());edit(data)
    path.write_text(json.dumps(data),encoding="utf-8")
    receipt=json.loads((pilot/"execution.json").read_text());receipt["artifacts"][name]=digest(path)
    (pilot/"execution.json").write_text(json.dumps(receipt),encoding="utf-8")


def test_offline_audit_accepts_preserved_history(copied_pilot):
    root,pilot=copied_pilot
    assert audit_pilot(root,pilot)["historical_attempts"]==3


@pytest.mark.parametrize("name,edit",[
    ("summary.json",lambda x:x.update(retention=.7)),
    ("summary.json",lambda x:x.update(controls_passed=True)),
    ("summary.json",lambda x:x.update(mean=.1)),
    ("summary.json",lambda x:x.update(selected_head=[0,0])),
    ("gate.json",lambda x:x.update(status="eligible_for_discovery")),
    ("membership.json",lambda x:x.update(biological_effects_used_for_selection=True)),
    ("confirmation_readiness.json",lambda x:x.update(ready=True)),
    ("eligibility_diagnostics.json",lambda x:x[1].update(candidate_positions_checked=129)),
    ("biological_qc.json",lambda x:x[0].update(confirmation_eligible=True)),
])
def test_semantic_tampering_rejected_despite_updated_hash(copied_pilot,name,edit):
    root,pilot=copied_pilot;edit_artifact(pilot,name,edit)
    with pytest.raises(ValueError):audit_pilot(root,pilot)


def test_stopped_pilot_cannot_acquire_head_results(copied_pilot):
    root,pilot=copied_pilot;(pilot/"discovery_heads.json").write_text("[]")
    with pytest.raises(ValueError,match="head-search"):audit_pilot(root,pilot)


def test_required_evidence_cannot_be_removed_from_receipt(copied_pilot):
    root,pilot=copied_pilot;p=pilot/"execution.json";data=json.loads(p.read_text())
    del data["artifacts"]["scores.csv"];p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="missing"):audit_pilot(root,pilot)


def test_inventory_cannot_promote_aggregate_counts_to_eligible_loci(tmp_path):
    shutil.copyfile(ROOT/"results/variant_study/adastra_tf_metadata.json",tmp_path/"adastra_tf_metadata.json")
    table=pd.read_csv(ROOT/"results/variant_study/dataset_inventory.csv")
    table.loc[table.dataset=="ADASTRA Mabel v6.1","eligible_loci"]=80000
    table.to_csv(tmp_path/"dataset_inventory.csv",index=False)
    with pytest.raises(ValueError,match="independent loci"):audit_inventory(tmp_path)


@pytest.mark.parametrize("field,value",[("retention",float('nan')),("retention",2),("clusters",float('nan')),
    ("clusters",True),("controls_passed","true"),("controls_passed",1),("binding_rho",2),("mean","0.1")])
def test_gate_cannot_pass_malformed_values(field,value):
    summary=dict(controls_passed=True,retention=.75,clusters=9,mean=.002,ci_low=.001,binding_rho=.8,binding_ci_low=.2)
    summary[field]=value
    assert feasibility_gate(summary,VariantProtocol())["status"]=="stop"


@pytest.mark.parametrize("changes",[{"binding_ci_low":2},{"binding_ci_low":.9},{"mean":.8},{"ci_low":.003}])
def test_gate_rejects_impossible_divergence_or_intervals(changes):
    summary=dict(controls_passed=True,retention=.75,clusters=9,mean=.002,ci_low=.001,binding_rho=.8,binding_ci_low=.2)
    assert feasibility_gate({**summary,**changes},VariantProtocol())["status"]=="stop"


def test_two_runners_cannot_claim_the_same_protocol(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from tools.run_variant_pilot import freeze_protocol
    barrier=Barrier(2)
    def claim(value):
        barrier.wait()
        try:freeze_protocol(tmp_path/"protocol.json",dict(owner=value));return True
        except FileExistsError:return False
    with ThreadPoolExecutor(2) as pool:results=list(pool.map(claim,[1,2]))
    assert sorted(results)==[False,True]
    assert json.loads((tmp_path/"protocol.json").read_text())["owner"] in (1,2)


@pytest.mark.parametrize("changes",[{"bootstrap_samples":True},{"query_radius_bp":1.5},{"seed":-1},
    {"gc_tolerance":True},{"min_binding_rho":float('nan')},{"min_retention":float('inf')}])
def test_protocol_rejects_invalid_configuration(changes):
    with pytest.raises(ValueError):replace(VariantProtocol(),**changes)


@pytest.mark.parametrize("logits",[torch.tensor([]),torch.tensor([1.]),torch.ones(2,3)])
def test_divergence_rejects_non_vocabulary_vectors(logits):
    with pytest.raises(ValueError):js_divergence(logits,logits)


def test_statistics_reject_missing_clusters_and_nonfinite_power():
    with pytest.raises(ValueError):cluster_summary([1,2],[0,float('nan')])
    with pytest.raises(ValueError):cluster_summary([[1,2]],[0])
    with pytest.raises(ValueError):simulate_cluster_power(np.linspace(-1,1,8),float('nan'))
    with pytest.raises(ValueError):simulate_cluster_power(np.linspace(-1,1,8),.1,candidates=(16,16))


def test_separation_reports_missing_identity_and_build_correspondence():
    a=dict(genome_build="hg19",chrom="chr1",reference_sequence="ACGT",window_start_0based=10,window_end_0based=14)
    b={**a,"genome_build":"hg38","reference_sequence":"AAAA"}
    result=separation_audit([a],[b])
    assert not result["exact_and_coordinate_separation_passed"]
    assert "missing donor_id" in result["reasons"]
    assert any("coordinate correspondence" in r for r in result["reasons"])
    with pytest.raises(ValueError):sequence_identity("NNNN")


def test_evidence_paths_cannot_escape_their_root(tmp_path):
    with pytest.raises(ValueError):within(tmp_path,"../outside")


def test_genomic_coordinates_cannot_be_silently_truncated():
    row=pd.read_csv(ROOT/"results/native_followup/ctcf_allele_effects.csv").iloc[0].to_dict()
    row["position_1based"]+=.1
    with pytest.raises(ValueError,match="coordinates"):validate_variant(row)
