from collections import Counter
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from src.assay_alignment import validate_patch_alignment, exact_offsets, intervention_positions, base_attention_density
from src.assay_stats import genomic_clusters, paired_auc_difference, paired_head_inference, absolute_patching_summary
from src.controlled_edits import MotifDefinition, controlled_edit_pairs, edit_signature
from src.config import DEFAULT_CONFIG
from src.provenance import validate_resume, write_stage_receipt


class BaseTokenizer:
    def __call__(self, sequence, **kwargs):
        offsets=[(0,0)]+[(i,i+1) for i in range(len(sequence))]+[(0,0)]
        return dict(offset_mapping=offsets,input_ids=list(range(len(offsets))))


def test_equal_token_count_with_different_nucleotide_intervals_is_rejected():
    class Retokenizing:
        def __call__(self, sequence, **kwargs):
            spans=[(0,0),(0,2),(2,4),(0,0)] if sequence=='AAAA' else [(0,0),(0,1),(1,4),(0,0)]
            return dict(offset_mapping=spans,input_ids=[0,1,2,3])
    with pytest.raises(ValueError,match='Nucleotide boundaries'):
        validate_patch_alignment(Retokenizing(),'AAAA','AATT')
    result=validate_patch_alignment(Retokenizing(),'AAAA','AATT',[0,3])
    assert result['positions']==[0,3]


def test_scientific_offsets_reject_gaps_and_truncation():
    class Broken:
        def __call__(self, sequence, **kwargs):
            return dict(offset_mapping=[(0,1),(2,4)])
    with pytest.raises(ValueError,match='gaps'):
        exact_offsets(Broken(),'ACGT')
    class Truncated:
        def __call__(self, sequence, **kwargs):
            return dict(offset_mapping=[(0,2)])
    with pytest.raises(ValueError,match='entire'):
        exact_offsets(Truncated(),'ACGT')


def test_base_density_is_invariant_to_partitioning_uniform_base_attention():
    # A token receives the mass of its bases. Subdivision must not change density.
    first=[(0,1),(1,4)]
    second=[(0,1),(1,2),(2,4)]
    a=np.tile(np.array([1.,3.])/4,(2,1))
    b=np.tile(np.array([1.,1.,2.])/4,(3,1))
    assert np.isclose(base_attention_density(a,first,(1,3)),1.)
    assert np.isclose(base_attention_density(b,second,(1,3)),1.)


def test_position_controls_exclude_specials_and_preserve_random_budget():
    offsets=exact_offsets(BaseTokenizer(),'ACGT'*12)
    edit=intervention_positions(offsets,(12,18),'edit')
    random=intervention_positions(offsets,(12,18),'random',seed=1)
    assert len(edit)==len(random) and not set(edit)&set(random)
    assert not {0,len(offsets)-1}&set(intervention_positions(offsets,(12,18),'all'))
    with pytest.raises(ValueError,match='Unknown'):
        intervention_positions(offsets,(12,18),'invalid')


def test_controlled_motif_and_sham_edits_preserve_composition_and_transitions():
    sequence='ACGTACGTACGTTATAAAACGTACGTACGTACGT'
    motif=MotifDefinition('tata',pattern='TATA[AT]A')
    pairs,audit=controlled_edit_pairs(sequence,'chr1:0-34',motif,BaseTokenizer(),max_pairs=2)
    assert len(pairs)==2 and audit['retained_pairs']==2
    for pair in pairs:
        edit,sham=pair['motif'],pair['sham']
        assert Counter(sequence)==Counter(edit['corrupted_sequence'])==Counter(sham['corrupted_sequence'])
        assert edit_signature(edit['clean_subsequence'],edit['corrupted_subsequence'])==edit_signature(sham['clean_subsequence'],sham['corrupted_subsequence'])
        assert len(motif.hits(edit['corrupted_sequence']))<len(motif.hits(sequence))
        assert motif.hits(sham['corrupted_sequence'])==motif.hits(sequence)


def test_genomic_blocks_union_matched_loci_overlaps_and_reverse_complements():
    names=np.array([['chr1:90-110','chr2:5-10'],['chr1:105-120','chr3:5-10'],['chr4:1-6','chr5:1-6']])
    seq=np.array([['AACCG','GGTTT'],['AAAAA','CCCCC'],['CGGTT','TTTTT']])
    codes=genomic_clusters(names,seq,block_bp=100)
    assert len(set(codes))==1
    with pytest.raises(ValueError,match='Missing genomic'):
        genomic_clusters(['bad-id'])


def test_identical_predictions_have_zero_paired_difference():
    y=np.array([0,1,0,1,0,1]);score=np.array([.2,.8,.4,.7,.3,.6])
    result=paired_auc_difference(y,score,score,repetitions=30)
    assert result['difference']==result['ci_low']==result['ci_high']==0


def test_cluster_head_null_and_simultaneous_inference_are_valid():
    result=paired_head_inference(np.zeros((6,3)),[0,0,1,1,2,2],repetitions=99,bootstrap_samples=40)
    assert np.all(result['p']==1) and np.all(result['max_stat_p']==1)
    assert np.all(result['simultaneous_low']==0)
    rng=np.random.default_rng(2)
    result=paired_head_inference(rng.normal(size=(20,4)),repetitions=99,bootstrap_samples=40)
    assert np.all(result['max_stat_p']>=result['p'])


def test_absolute_effects_expose_ratio_outlier_and_leave_one_out_range():
    clean=np.array([.01,1.,1.]);corrupt=np.zeros(3);patched=np.array([[.1],[.1],[.1]])
    result=absolute_patching_summary(clean,corrupt,patched,repetitions=20)
    assert np.allclose(result['mean'],.1)
    assert np.allclose(result['loo_low'],.1) and np.allclose(result['loo_high'],.1)


@pytest.mark.parametrize('change',['revision','probe_layer','activation_layers'])
def test_resume_rejects_changed_scientific_settings(tmp_path,change):
    config=replace(DEFAULT_CONFIG,paths=replace(DEFAULT_CONFIG.paths,project_root=tmp_path,manifests_dir=tmp_path/'manifests'))
    artifact=tmp_path/'cache.npz';artifact.write_bytes(b'cache')
    write_stage_receipt('cache',config,dict(path=str(artifact)))
    if change=='revision':
        changed=replace(config,model=replace(config.model,revision='other-checkpoint'))
    else:
        changed=replace(config,data=replace(config.data,**{change:5 if change=='probe_layer' else (5,)}))
    with pytest.raises(ValueError,match='configuration changed'):
        validate_resume(changed,['cache'])


def test_resume_rejects_corrupted_artifact_and_missing_receipt(tmp_path):
    config=replace(DEFAULT_CONFIG,paths=replace(DEFAULT_CONFIG.paths,project_root=tmp_path,manifests_dir=tmp_path/'manifests'))
    artifact=tmp_path/'cache.npz';artifact.write_bytes(b'cache')
    write_stage_receipt('cache',config,dict(path=str(artifact)))
    assert len(validate_resume(config,['cache']))==1
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):
        validate_resume(config,['cache'])
    with pytest.raises(ValueError,match='without a verified'):
        validate_resume(config,['absent'])


def test_patching_writer_preserves_noncontiguous_layer_ids(tmp_path):
    from tests.test_counterfactuals_and_patching import tmp_config
    from src.patching import save_restoration_matrix
    config=tmp_config(tmp_path)
    outputs=save_restoration_matrix(np.ones((2,2)),'subset',config,layer_indices=(5,11))
    assert pd.read_csv(outputs['table']).layer.tolist()==[5,5,11,11]


def test_strict_geometry_matching_excludes_unequal_real_token_counts():
    from tools.review_ctcf_controls import match_controls
    sequences=pd.DataFrame(dict(sequence=['ACGT'*4]*2,chrom=['chr1']*2))
    tokens=pd.DataFrame(dict(sequence_index=[0,0,1,1,1,1],token=[0,1,0,1,2,3],
                            char_start=[0,8,0,4,8,12],char_end=[8,16,4,8,12,16],
                            is_support=[True,False,False,False,False,False]))
    assert len(match_controls(sequences,tokens))==1
    assert match_controls(sequences,tokens,strict_geometry=True)==[]


def test_known_fixture_distinguishes_detector_from_composition_control():
    from src.assay_calibration import KnownAttentionFixture
    motif=MotifDefinition('tata',pattern='TATAAA')
    clean='ACGT'*4+'TATAAA'+'ACGT'*4
    corrupt=clean.replace('TATAAA','AAATAT')
    detector=KnownAttentionFixture('single',2.,1)
    control=KnownAttentionFixture('composition',2.,1)
    assert detector.forward(clean,motif)['score']>detector.forward(corrupt,motif)['score']
    assert np.isclose(control.forward(clean,motif)['score'],control.forward(corrupt,motif)['score'])

def test_single_pair_absolute_summary_withholds_uncertainty():
    result=absolute_patching_summary([1.],[0.],[[.25]])
    assert result["mean"][0]==.25 and np.isnan(result["ci_low"][0])
    assert np.isnan(result["loo_low"][0])


def test_resume_rejects_modified_pooling_receipt(tmp_path):
    import hashlib
    config=replace(DEFAULT_CONFIG,paths=replace(DEFAULT_CONFIG.paths,project_root=tmp_path,manifests_dir=tmp_path/"manifests"))
    artifact=tmp_path/"cache.npz";artifact.write_bytes(b"cache")
    path=write_stage_receipt("cache",config,dict(path=str(artifact)))
    receipt=json.loads(path.read_text())
    receipt["scientific_config"]["pooling"]="first token"
    receipt["fingerprint"]=hashlib.sha256(json.dumps(receipt["scientific_config"],sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match="configuration changed"):
        validate_resume(config,["cache"])


def test_incremental_models_share_population_and_freeze_validation_choices(tmp_path):
    from src.incremental_prediction import compare_readouts, classifier_digest
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(43)
    def population(chrom,n):
        y=np.arange(n)%2
        x=np.column_stack([y+rng.normal(0,.1,n),rng.normal(size=n)])
        seq=np.asarray(["".join(rng.choice(list("ACGT"),32)) for _ in range(n)])
        names=np.asarray([f"chr{chrom}:{i*2000000}-{i*2000000+32}" for i in range(n)])
        return x,y,names,seq
    train,val,test=population(1,24),population(18,16),population(20,16)
    with threadpool_limits(1):
        scorer,receipt=compare_readouts("fixture",train,val,test,tmp_path,bootstrap_samples=30)
    prediction=pd.read_csv(tmp_path/"fixture_predictions.csv")
    assert prediction.name.tolist()==test[2].tolist()
    assert len(receipt["classifier_selection"])==5
    assert receipt["classifier_selection"]["residual"]["classifier_sha256"]==classifier_digest(scorer)
    assert receipt["partitions"]==[["chr1"],["chr18"],["chr20"]]
    comparisons=pd.read_csv(tmp_path/"fixture_paired_comparisons.csv")
    assert len(comparisons)==6 and np.isfinite(comparisons.ci_low).all()
    with pytest.raises(ValueError,match="chromosomes"):
        compare_readouts("bad",train,val,train,tmp_path)


def test_interval_overlap_handles_nested_and_touching_windows():
    from src.ctcf_controls import interval_index, overlaps_any
    index=interval_index(pd.DataFrame(dict(chrom=["chr1"]*3,start=[10,20,80],end=[100,25,90])))
    assert overlaps_any("chr1",60,70,index)
    assert not overlaps_any("chr1",100,110,index)
    assert not overlaps_any("chr2",0,10,index)


def test_calibration_exposes_query_specific_edit_only_blindness(tmp_path):
    from src.assay_calibration import run_calibration
    manifest=run_calibration(tmp_path,repetitions=4)
    saved=pd.read_csv(tmp_path/"calibration_observations.csv")
    query=saved[saved.kind=="query_specific"]
    assert manifest["confirmation_false_positive_rate"]<=.05
    assert (query.absolute_effect>0).all()
    assert np.allclose(query.loc[query.position<88,"edit_only_effect"],0)
    assert (query.edit_only_effect==0).mean()>.8
    assert set(saved.partition)=={"discovery","confirmation"}


def test_manuscript_build_rejects_changed_execution_input(tmp_path):
    from tools.build_hardened_artifacts import validate_executions
    from tools.run_hardened_assay import STAGES
    from src.utils import sha256_file
    output=tmp_path/"out";output.mkdir()
    (output/"protocol.json").write_text("{}")
    source=tmp_path/"science.py";source.write_text("valid")
    artifact=output/"input.csv";artifact.write_text("valid")
    for stage in STAGES:
        folder=output/stage;folder.mkdir()
        (folder/"execution.json").write_text(json.dumps(dict(status="completed",
            protocol_sha256=sha256_file(output/"protocol.json"),source_changed_during_stage=[],
            source_sha256={"science.py":sha256_file(source)},artifacts={"input.csv":sha256_file(artifact)})))
    validate_executions(output,tmp_path)
    artifact.write_text("changed")
    with pytest.raises(ValueError,match="Changed stage artifact"):
        validate_executions(output,tmp_path)


@pytest.mark.artifact
def test_hardened_manuscript_artifacts_recompute_and_verify():
    from pathlib import Path
    from src.utils import sha256_file
    root=Path(__file__).resolve().parents[1]
    output=root/"results/hardened"
    manifest=json.loads((output/"paper_manifest.json").read_text())
    assert sha256_file(root/manifest["paper"]["path"])==manifest["paper"]["sha256"]
    for name,digest in {**manifest["sources"],**manifest["figures"]}.items():
        assert sha256_file(root/name)==digest
    for folder in [output/"patching/promoter_tata",output/"ctcf/interventions"]:
        effects=pd.read_csv(folder/"confirmation_pair_effects.csv")
        summary=pd.read_csv(folder/"confirmation_summary.csv")
        assert np.allclose(effects.absolute_effect,effects.patched_score-np.where(
            effects.direction=="denoise",effects.corrupted_score,effects.clean_score))
        for row in summary.itertuples():
            subset=effects[(effects.scheme==row.scheme)&(effects.direction==row.direction)]
            pivot=subset.pivot(index=["sequence_id","candidate_index"],columns="role",values="absolute_effect")
            means=(pivot.motif-pivot.sham).groupby(level="sequence_id").mean()
            assert np.isclose(means.mean(),row.mean_absolute_motif_minus_sham)
            assert means.size==row.sequences


def test_reused_cache_requires_verified_bytes_and_checkpoint(tmp_path):
    from src.provenance import verify_reproduced_caches
    from src.utils import sha256_file
    train=tmp_path/"train.npz";test=tmp_path/"test.npz"
    train.write_bytes(b"training");test.write_bytes(b"testing")
    comparison=tmp_path/"comparison.json";reproduction=tmp_path/"reproduction.json"
    comparison.write_text(json.dumps(dict(records=[dict(task="fixture",identity=dict(names=True,sequences=True,labels=True,layers=True),
        maximum_absolute_feature_difference=0,new_sha256=sha256_file(test),old_sha256=sha256_file(test),
        training_cache_sha256=dict(old=sha256_file(train),new=sha256_file(train)))])))
    reproduction.write_text(json.dumps(dict(exit_status=0,model_revision=DEFAULT_CONFIG.model.revision,
        dataset_revision=DEFAULT_CONFIG.data.hf_dataset_revision)))
    verify_reproduced_caches([train,test],"fixture",comparison,reproduction,DEFAULT_CONFIG)
    incompatible=replace(DEFAULT_CONFIG,model=replace(DEFAULT_CONFIG.model,revision="different"))
    with pytest.raises(ValueError,match="incompatible checkpoint"):
        verify_reproduced_caches([train,test],"fixture",comparison,reproduction,incompatible)
    train.write_bytes(b"changed")
    with pytest.raises(ValueError,match="features changed"):
        verify_reproduced_caches([train,test],"fixture",comparison,reproduction,DEFAULT_CONFIG)
