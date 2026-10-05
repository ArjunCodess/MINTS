import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.native_alleles import allele_effect,extract_alleles
from src.native_followup import (target_geometry,match_target_geometry,recovery_targets,
    engineered_motif_fixture,influence_table,target_logits)
from src.utils import sha256_file


def test_same_side_disjoint_windows_cannot_match_tighter_than_their_width():
    target=(60,65)
    motif=(30,49)
    for start in range(42):
        sham=(start,start+19)
        if sham[1]<=motif[0] or sham[0]>=motif[1]:
            assert not match_target_geometry(motif,sham,target,2)
    assert match_target_geometry((30,32),(27,29),target,3)
    assert not match_target_geometry((30,32),(30,32),target,0)
    assert not match_target_geometry((30,32),(66,68),target,40)
    assert target_geometry((60,64),target)[0]=="overlap"
    with pytest.raises(ValueError):
        match_target_geometry(motif,(0,19),target,-1)


def test_recovery_sampling_is_fixed_and_excludes_pilot_chromosomes():
    class Tokenizer:
        mask_token_id=9
        all_special_ids=[0,9]
        def __call__(self,sequence,**kwargs):
            return dict(input_ids=[0]+["ACGT".index(b)+1 for b in sequence]+[0])
    # exact_offsets calls the tokenizer with an offsets request.
    class WithOffsets(Tokenizer):
        def __call__(self,sequence,**kwargs):
            return dict(**super().__call__(sequence,**kwargs),
                        offset_mapping=[(0,0)]+[(i,i+1) for i in range(len(sequence))]+[(0,0)])
    table=pd.DataFrame(dict(chrom=["chr16","chr17","chr18"],start=[0,10,20],end=[4,14,24],sequence=["ACGT"]*3))
    first=recovery_targets(WithOffsets(),table,["chr16","chr17"],2,1730)
    assert first==recovery_targets(WithOffsets(),table,["chr16","chr17"],2,1730)
    assert all(x["sequence_id"].split(":")[0] in ["chr16","chr17"] for x in first)
    assert all(x["masked_ids"][x["target"]["index"]]==9 for x in first)


def test_engineered_motif_fixture_has_signal_and_exact_rescue():
    result=engineered_motif_fixture()
    assert result["passed"] and result["clean_minus_edited_log_probability"]>.5
    assert result["rescue_max_logit_error"]==0


def test_hooks_are_removed_after_forward_failure():
    context=torch.nn.Identity()
    layer=torch.nn.Identity()
    layer.attention=SimpleNamespace(self=context)
    class Broken:
        bert=SimpleNamespace(encoder=SimpleNamespace(layer=[layer]))
        def __call__(self,**kwargs):
            raise RuntimeError("failed forward")
    bundle=SimpleNamespace(hf_model=Broken(),device="cpu")
    with pytest.raises(RuntimeError,match="failed forward"):
        target_logits(bundle,[0,9,0],dict(index=1,token_id=1),cache=True)
    assert not context._forward_hooks and not layer._forward_hooks


def test_pooled_count_normalization_matches_publication_and_handles_zeros():
    result=allele_effect(14,5,29,3)
    assert result["normalized_vaf"]==pytest.approx(8.4/37.4)
    assert result["log_odds_ratio"]<0
    assert np.isfinite(allele_effect(17,16,18,0)["ci_low"])
    assert allele_effect(0,13,0,14)["status"]=="not_identifiable"
    with pytest.raises(ValueError):
        allele_effect(-1,1,1,1)


def test_parser_groups_adjacent_variants_and_rejects_wrong_counts():
    text="Table S3: COLO829\nchr2:43,158,779 C>T 1 4 14 5 29 3 22% heterozygous\nchr2:43,158,780 C>T 2 5 14 5 29 3 22% heterozygous\nWGS = whole-genome"
    rows=extract_alleles(text)
    assert rows[0]["locus_group"]==rows[1]["locus_group"]
    assert rows[0]["position_1based"]==43158779
    with pytest.raises(ValueError,match="rounded"):
        extract_alleles(text.replace("22%","80%"))


def test_influence_removes_whole_clusters():
    scores=pd.DataFrame(dict(sequence_id=["chr18:0-4","chr18:8-12","chr19:0-4"],
                             clean_sequence=["AAAA","AAAT","TTTC"],motif_minus_sham_loss=[1.,3.,-2.]))
    result=influence_table(scores)
    assert list(result.leave_cluster_out_mean)==[-2.,-2.,2.]


@pytest.mark.artifact
def test_followup_receipts_and_source_table_are_consistent():
    root=Path(__file__).resolve().parents[1]
    output=root/"results/native_followup"
    receipt=json.loads((output/"execution.json").read_text())
    assert receipt["status"]=="completed" and not receipt["source_changed_during_run"]
    for name,digest in receipt["source_sha256"].items():
        assert sha256_file(root/name)==digest
    for name,digest in receipt["artifacts"].items():
        assert sha256_file(output/name)==digest
    controls=pd.read_csv(output/"intervention_controls.csv")
    assert len(controls)==24 and controls.passed.all()
    recovery=pd.read_csv(output/"recovery_scores.csv")
    assert len(recovery)==128
    assert all(n.split(":")[0] in {"chr16","chr17"} for n in recovery.sequence_id)
    manifest=json.loads((output/"allele_manifest.json").read_text())
    assert manifest["table_sha256"]==sha256_file(output/"ctcf_allele_effects.csv")
    assert manifest["parser_sha256"]==sha256_file(root/"src/native_alleles.py")
    alleles=pd.read_csv(output/"ctcf_allele_effects.csv")
    assert len(alleles)==16 and alleles.reference_verified.all()
    assert (alleles.status!="not_identifiable").sum()==15
    assert manifest["heterozygous_loci"]==13
    assert alleles.groupby("locus_group").size().gt(1).sum()==2
    report=json.loads((output/"report_manifest.json").read_text())
    assert report["paper_tex_sha256"]==sha256_file(root/"paper/native_followup.tex")
    assert report["generator_sha256"]==sha256_file(root/"tools/build_native_followup.py")
    for name,digest in report["outputs"].items():
        assert sha256_file(output/name)==digest
