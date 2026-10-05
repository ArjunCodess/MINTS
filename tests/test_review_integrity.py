import numpy as np
import pandas as pd
import pytest
from src.integrity import assert_disjoint_splits, near_duplicate_pairs
from src.inference import holm_adjust, validate_metric
from src.probing import _matched_gc_indices
from tools.review_audit import count_stages
from tools.review_tables import tex_table
from tools.review_ctcf_controls import match_controls, paired_inference, sequence_qk_correlation


def row(sequence, name, label=1):
    return dict(sequence=sequence, name=name, label=label)


def test_leakage_checks_detect_reverse_complement_and_interval_overlap():
    with pytest.raises(ValueError, match="duplication"):
        assert_disjoint_splits(dict(train=[row("AACCGG", "chr1:0-6")],test=[row("CCGGTT", "chr2:0-6")]))
    with pytest.raises(ValueError, match="overlapping"):
        assert_disjoint_splits(dict(train=[row("AACCGG", "chr1:0-6")],test=[row("AAAAAA", "chr1:5-11")]))
    assert_disjoint_splits(dict(train=[row("AACCGG", "chr1:0-6")],test=[row("AAAAAA", "chr2:5-11")]))


def test_near_duplicates_detect_substitutions_in_either_orientation():
    result = near_duplicate_pairs(dict(train=[row("AACCGGTTAA", "chr1:0-10")],test=[row("AACCGGTTAT", "chr2:0-10")]))
    assert result[0]["distance"] == 1


def test_matching_does_not_reorder_an_imbalanced_gc_distribution_as_a_control():
    sequences = np.array(["AAAA", "AAAT", "GCCC", "GGGC"])
    labels = np.array([1,1,0,0])
    assert len(_matched_gc_indices(sequences,labels,1729,caliper=None)) == 4
    assert len(_matched_gc_indices(sequences,labels,1729,caliper=0.02)) == 0


def test_unique_tokens_and_hit_occurrences_are_distinct():
    table = pd.DataFrame(dict(is_support=[True,True,False],motif_score=[2.,3.,np.nan],threshold=[1.,1.,1.],
                              support_span_count=[2,1,0],sequence_index=[0,0,1]))
    stages = count_stages(table)
    assert stages["unique_motif_support_tokens"] == 2
    assert stages["motif_hit_token_occurrences"] == 3
    assert stages["motif_absent_sequences"] == 1


def test_metric_ranges_distinguish_ratio_from_correlation():
    validate_metric("attention_enrichment_ratio", [1.31])
    with pytest.raises(ValueError):
        validate_metric("spearman_rho", [1.31])
    with pytest.raises(ValueError):
        validate_metric("auroc", [1.01])
    assert np.allclose(holm_adjust([.01,.04,.03]), [.03,.06,.06])


def test_numeric_table_generation():
    rows=pd.DataFrame([dict(task="promoter_tata",method="probe",auroc=.7,auroc_ci_low=.6,auroc_ci_high=.8)])
    assert "0.7000 [0.6000, 0.8000]" in tex_table(rows,dict(probe="Readout"))


def test_genomic_controls_match_chromosome_gc_and_length_without_replacement():
    sequences=pd.DataFrame(dict(sequence=["ACGT"*10,"ACGT"*10,"ACGT"*10,"ACGT"*10],chrom=["chr1","chr2","chr1","chr3"]))
    tokens=pd.DataFrame(dict(sequence_index=[0,1,2,3],is_support=[True,True,False,False]))
    pairs=match_controls(sequences,tokens)
    assert len(pairs)==1
    assert pairs[0]["present_index"]==0 and pairs[0]["absent_index"]==2


def test_genomic_controls_reject_motif_scores_from_reordered_sequences():
    sequences=pd.DataFrame(dict(sequence=["ACGT"*10]*2,chrom=["chr1"]*2,name=["a","b"]))
    tokens=pd.DataFrame(dict(sequence_index=[0,1],sequence_id=["b","a"],is_support=[True,False]))
    with pytest.raises(ValueError,match="sequence IDs"):
        match_controls(sequences,tokens)


def test_sequence_pair_null_does_not_manufacture_signal_from_zero_differences():
    effect,interval,p,adjusted=paired_inference(np.zeros((4,2)),repetitions=99)
    assert np.all(effect==0) and np.all(interval==0)
    assert np.all(p==1) and np.all(adjusted==1)


def test_repeated_sequences_are_resampled_as_one_cluster():
    from src.inference import bootstrap_sequence_cluster_interval, bootstrap_mean_interval
    values=np.array([0.,10.,10.,10.])
    assert bootstrap_sequence_cluster_interval(values,["a","b","b","b"])==[0.,10.]
    assert bootstrap_mean_interval(values)[0]>0
    assert bootstrap_sequence_cluster_interval(values,["a","b","c","d"])==bootstrap_mean_interval(values)


def test_sequence_qk_ignores_missing_tokens_and_preserves_correlation_bounds():
    y=np.array([1.,2.,3.,np.nan])
    r=sequence_qk_correlation([[1.,2.,3.,99.],[3.,2.,1.,99.],[1.,1.,1.,99.]],y)
    assert np.allclose(r[:2],[1.,-1.]) and np.isnan(r[2])
    assert np.isnan(sequence_qk_correlation([[1.,2.,3.]], [0.,0.,0.])).all()


def test_manuscript_sources_have_no_stale_submission_material():
    from pathlib import Path
    import re
    root=Path(__file__).resolve().parents[1]
    text="\n".join(p.read_text(encoding="utf-8") for p in (root/"paper").glob("*.tex"))
    assert not re.search(r"rebuttal|reviewer\s*[123]|iclr|icml|preregistered",text,re.I)
    assert not re.search(r"\\label\{(?:tab:)?(?:table)?[89]\}",text,re.I)
    active_code="\n".join(p.read_text(encoding="utf-8") for folder in ["src","tools"] for p in (root/folder).glob("*.py"))
    assert "suggested_by" not in active_code


@pytest.mark.artifact
def test_paper_numbers_and_inputs_match_the_saved_artifact_manifest():
    from pathlib import Path
    import hashlib
    import json
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/"results/review/tables_manifest.json").read_text())
    sources={**manifest["paper_sources"],manifest["paper_results"]["path"]:manifest["paper_results"]["sha256"]}
    for path,expected in sources.items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected, f"Stale manuscript source: {path}"
    for path,record in manifest["figures"].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==record["sha256"], f"Stale figure: {path}"
        assert all(source in manifest["paper_sources"] for source in record["sources"])


def test_manuscript_import_rejects_an_unfinished_run(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/"tools/review_manuscript_artifacts.py"),
                           "--input-run-directory",str(tmp_path)],capture_output=True,text=True,timeout=30)
    assert result.returncode!=0
    assert "Cannot import incomplete scientific run" in result.stderr


@pytest.mark.artifact
def test_reported_classifier_metrics_recompute_from_saved_predictions():
    from pathlib import Path
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
    out=Path(__file__).resolve().parents[1]/"results/review"
    metrics=pd.read_csv(out/"classification_metrics.csv")
    for task,rows in metrics.groupby("task"):
        predictions=pd.read_csv(out/f"{task}_predictions.csv")
        for row in rows.itertuples():
            subset=predictions
            if row.method=="historical_gc_probe":
                subset=subset[subset.historical_gc_selected]
            elif row.method.startswith("caliper_gc_"):
                subset=subset[subset.gc_caliper_selected]
            column={"gc":"gc_probability", "kmer":"kmer_probability",
                    "caliper_gc_baseline":"gc_probability"}.get(row.method,"probe_probability")
            y,score=subset.label.to_numpy(),subset[column].to_numpy()
            assert len(y)==row.test_examples
            assert np.isclose(roc_auc_score(y,score),row.auroc,atol=1e-12,rtol=0)
            assert np.isclose(average_precision_score(y,score),row.auprc,atol=1e-12,rtol=0)
            assert np.isclose(accuracy_score(y,score>=.5),row.accuracy,atol=1e-12,rtol=0)


@pytest.mark.artifact
def test_native_and_qk_summaries_match_raw_sequence_pairs():
    from pathlib import Path
    out=Path(__file__).resolve().parents[1]/"results/review"
    with np.load(out/"ctcf_native_control_scores.npz") as saved:
        for filename,key in [("ctcf_native_control_inference.csv","scores"),
                             ("ctcf_sequence_qk_inference.csv","qk_correlations")]:
            values=saved[key].astype(float)
            values=values[np.isfinite(values).all(axis=(1,2,3))]
            table=pd.read_csv(out/filename).sort_values(["layer","head"])
            means=(values[:,0]-values[:,1]).mean(axis=0).reshape(-1)
            assert len(table)==144 and (table.pairs==len(values)).all()
            # Native scores were differenced in float32 before inference;
            # recomputation in float64 can differ by that rounding precision.
            assert np.allclose(means,table.mean_present_minus_absent,atol=5e-8,rtol=0)
            if key=="qk_correlations":
                assert np.allclose(values[:,0].mean(axis=0).reshape(-1),table.mean_present_r,atol=1e-12,rtol=0)
            p=table.permutation_p.to_numpy()
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            order=np.argsort(p)
            expected=np.asarray([min(1.,max((len(p)-j)*p[order[j]] for j in range(i+1)))
                                 for i in range(len(p))])
            assert np.allclose(table.holm_p.to_numpy()[order],expected,atol=1e-12,rtol=0)


@pytest.mark.artifact
def test_heldout_patching_interpretation_matches_raw_scores():
    from pathlib import Path
    import json
    root=Path(__file__).resolve().parents[1]
    patchroot=root/"results/cross_model/review_tata_heldout/patching"
    stem="promoter_tata_batch_dnabert_activation_patching"
    with np.load(patchroot/(stem+"_pair_effects.npz")) as saved:
        assert all(str(identity).startswith("test:") for identity in saved["sequence_ids"])
        denominator=saved["denominator"]
        assert np.allclose(denominator,saved["clean_scores"]-saved["corrupted_scores"])
        expected=(saved["patched_scores"]-saved["corrupted_scores"][:,None,None])/denominator[:,None,None]
        assert np.allclose(saved["restoration"],expected,atol=1e-12,rtol=0)
        table=pd.read_csv(patchroot/(stem+".csv"))
        for row in table.itertuples():
            values=expected[:,row.layer,row.head]
            assert np.isclose(row.restoration,values.mean(),atol=1e-12,rtol=0)
            assert np.isclose(row.median_restoration,np.median(values),atol=1e-12,rtol=0)
        summary=json.loads((root/"results/review/heldout_patching_summary.json").read_text())
        assert summary["pm_greater_than_one"]==int((expected>1).sum())
        assert summary["negative_denominators"]==int((denominator<0).sum())
        assert summary["positive_denominators"]==int((denominator>0).sum())
