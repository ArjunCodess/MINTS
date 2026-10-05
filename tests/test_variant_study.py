"""Meaningful safety, alignment, statistics and scientific receipt regressions."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.controlled_edits import MotifDefinition
from src.variant_assay import validate_variant, prepare_variant, js_divergence, score_variant
from src.variant_protocol import VariantProtocol, feasibility_gate, confirmation_readiness
from src.variant_quality import biological_qc, separation_audit
from src.variant_statistics import cluster_summary, binding_agreement, simulate_cluster_power, select_head
from src.utils import sha256_file


class CharacterTokenizer:
    mask_token_id=9
    def __call__(self,sequence,**kwargs):
        return dict(input_ids=[10]+["ACGT".index(x)+1 for x in sequence]+[11],
                    offset_mapping=[(0,0)]+[(i,i+1) for i in range(len(sequence))]+[(0,0)])


def variant():
    sequence="T"*20+"ACGA"+"TTTCGA"+"T"*50
    index=22
    return dict(genome_build="hg19",position_1based=1023,window_start_0based=1000,
        window_end_0based=1000+len(sequence),reference="G",alternate="T",reference_sequence=sequence,
        alternate_sequence=sequence[:index]+"T"+sequence[index+1:],reference_verified=True,
        wgs_wt=20,wgs_mt=20,chip_wt=30,chip_mt=10,zygosity="heterozygous",chrom="chr1",locus_group="locus")


def test_coordinate_allele_and_count_validation():
    row=variant();assert validate_variant(row)==22
    for changes in ({"position_1based":1024},{"reference":"A"},{"alternate_sequence":row["reference_sequence"]},
                    {"reference_verified":False},{"genome_build":"unknown"},{"chip_mt":-1},{"chip_mt":.5}):
        with pytest.raises(ValueError):validate_variant({**row,**changes})


def test_control_keeps_edit_visible_and_geometry_aligned():
    p=replace(VariantProtocol(),gc_tolerance=.2)
    case,reason=prepare_variant(CharacterTokenizer(),variant(),MotifDefinition("CTCF",pattern="ACGA",scan_reverse=False),p,{"locus":1})
    assert reason=="retained"
    assert case["sham_index"]==28
    assert case["trinucleotide"]=="CGA"
    assert case["substitution"]=="G>T"
    for q in case["queries"]:
        assert not q["span"][0]<=22<q["span"][1]
        assert not q["span"][0]<=28<q["span"][1]
        a,b,c=[ids.copy() for ids in case["ids"]]
        for ids in (a,b,c):ids[q["index"]]=9
        assert a!=b and a!=c


def test_unresolved_phase_and_homozygous_excluded_before_scoring():
    args=(CharacterTokenizer(),variant(),MotifDefinition("CTCF",pattern="ACGA"),VariantProtocol())
    assert prepare_variant(*args,{"locus":2})[1]=="adjacent variants with unresolved phase"
    assert prepare_variant(args[0],{**args[1],"zygosity":"homozygous"},*args[2:],{"locus":1})[0] is None


def test_bpe_boundary_changes_rejected():
    class VariableTokenizer(CharacterTokenizer):
        def __call__(self,sequence,**kwargs):
            value=super().__call__(sequence,**kwargs)
            if "ACTA" in sequence:
                value["offset_mapping"][23]=(22,24)
                del value["offset_mapping"][24];del value["input_ids"][24]
            return value
    assert prepare_variant(VariableTokenizer(),variant(),MotifDefinition("CTCF",pattern="ACGA"),VariantProtocol(),{"locus":1})[1]=="reference/alternate BPE boundaries differ"


def test_native_distribution_divergence_identity_symmetry_and_bound():
    a=torch.tensor([3.,-2.,1.]);b=torch.tensor([-1.,2.,1.])
    assert js_divergence(a,a)==pytest.approx(0,abs=1e-12)
    value=js_divergence(a,b)
    assert 0<value<np.log(2)
    assert value==pytest.approx(js_divergence(b,a))
    assert value==pytest.approx(js_divergence(a+100,b-100))
    with pytest.raises(ValueError):js_divergence(a,torch.tensor([float('nan')]*3))


def test_native_scoring_and_one_head_rescue_use_actual_hooks():
    from types import SimpleNamespace
    class Context(torch.nn.Module):
        attention_head_size=1
        num_attention_heads=2
        def forward(self,ids):
            values=torch.zeros(len(ids),2)
            values[:,0]=float(ids[1]==1)
            return values
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__();self.context=Context();self.attention=SimpleNamespace(self=self.context)
        def forward(self,ids):return self.context(ids)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.layer=Layer();self.bert=SimpleNamespace(encoder=SimpleNamespace(layer=[self.layer]))
        def forward(self,input_ids,**kwargs):
            context=self.layer(input_ids[0]);return SimpleNamespace(logits=torch.stack([context[:,1],4*context[:,0]],-1)[None])
    model=Model();bundle=SimpleNamespace(hf_model=model,tokenizer=CharacterTokenizer(),device="cpu")
    case=dict(queries=[dict(index=4,span=[3,4],width=1,token_id=1)],ids=[[0,1,2,3,1,0],[0,2,2,3,1,0],[0,1,3,3,1,0]])
    score=score_variant(bundle,case,VariantProtocol(),head=(0,0))
    assert score["variant_divergence"]>.1 and score["sham_divergence"]==pytest.approx(0,abs=1e-12)
    assert score["head_contrast"]==pytest.approx(score["contrast"])
    assert score["controls_passed"]
    assert not model.layer.context._forward_hooks and not model.layer._forward_hooks


def test_equal_cluster_weighting_prevents_sequence_pseudoreplication():
    result=cluster_summary([1]*10+[3],[0]*10+[1],repetitions=200,seed=4)
    assert result["mean"]==2 and result["clusters"]==2
    assert cluster_summary([1],[0])["mean"] is None
    assert binding_agreement([1,1,1],[1,2,3],[0,1,2])["binding_rho"] is None


def test_pilot_gate_stops_on_missing_and_null_evidence():
    p=VariantProtocol()
    assert feasibility_gate({},p)["status"]=="stop"
    valid=dict(controls_passed=True,retention=.75,clusters=9,mean=.002,ci_low=.001,
               binding_rho=.8,binding_ci_low=.2)
    assert feasibility_gate(valid,p)["status"]=="eligible_for_discovery"
    for key,value in (("controls_passed",False),("retention",.1),("clusters",2),("mean",float('nan')),
                      ("ci_low",0),("binding_rho",.1),("binding_ci_low",None)):
        assert feasibility_gate({**valid,key:value},p)["status"]=="stop"


def test_confirmation_requires_every_named_requirement():
    assert not confirmation_readiness({"pilot_passed":True})["ready"]
    required=confirmation_readiness({})["missing"]
    assert confirmation_readiness({k:True for k in required})["ready"]
    assert not confirmation_readiness({k:"true" for k in required})["ready"]
    assert not biological_qc(variant())["confirmation_eligible"]


def test_source_donor_and_reverse_complement_separation():
    row=variant();row.update(donor_id="person",source_experiment="experiment")
    other={**row,"chrom":"chr2","reference_sequence":row["reference_sequence"].translate(str.maketrans("ACGT","TGCA"))[::-1]}
    result=separation_audit([row],[other])
    assert "exact/reverse-complement sequence overlap" in result["reasons"]
    assert "shared donor_id" in result["reasons"]
    assert not result["verified"]


def test_intervention_power_requires_real_cluster_noise_and_retention():
    with pytest.raises(ValueError):simulate_cluster_power([.1]*8,.001)
    with pytest.raises(ValueError):simulate_cluster_power([.1,.2],.001)
    result=simulate_cluster_power(np.linspace(-.002,.002,12),.005,retention=.5,candidates=(16,),repetitions=200)
    assert result["selected"]["retained_clusters"]==16
    assert result["selected"]["screen_clusters"]==32
    assert select_head([dict(layer=1,head=0,mean=1),dict(layer=0,head=2,mean=1)])["layer"]==0


@pytest.mark.artifact
def test_variant_execution_and_baseline_receipts():
    root=Path(__file__).resolve().parents[1]
    folder=root/"results/variant_pilot"
    receipt=json.loads((folder/"execution.json").read_text())
    assert receipt["status"]=="completed" and not receipt["source_changed_during_run"]
    for name,digest in receipt["source_sha256"].items():assert sha256_file(root/name)==digest
    for name,digest in receipt["artifacts"].items():assert sha256_file(folder/name)==digest
    baseline=json.loads((root/"results/variant_study/baseline_manifest.json").read_text())
    resolved=__import__('subprocess').check_output(['git','rev-parse',baseline['commit']+'^{commit}'],cwd=root,text=True).strip()
    assert baseline["commit"]==resolved
    assert not json.loads((folder/"confirmation_readiness.json").read_text())["ready"]
    report=json.loads((root/"results/variant_study/report_manifest.json").read_text())
    assert report["generator_sha256"]==sha256_file(root/"tools/build_variant_artifacts.py")
    for name,digest in report["outputs"].items():assert sha256_file(root/name)==digest
