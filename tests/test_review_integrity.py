import numpy as np
import pandas as pd
import pytest
from src.integrity import assert_disjoint_splits, near_duplicate_pairs
from src.inference import holm_adjust, validate_metric
from src.probing import _matched_gc_indices
from tools.review_audit import count_stages
from tools.review_submission import stale_findings
from tools.review_tables import tex_table
from tools.review_ctcf_controls import match_controls, paired_inference


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


def test_submission_remnants_and_numeric_generation():
    assert stale_findings("Response-to-Review Checklist: mV6D. preregistered thresholds")
    assert not stale_findings("No head passed the motif-local screens; thresholds were heuristic.")
    rows=pd.DataFrame([dict(task="promoter_tata",method="probe",auroc=.7,auroc_ci_low=.6,auroc_ci_high=.8)])
    assert "0.7000 [0.6000, 0.8000]" in tex_table(rows,dict(probe="Readout"))


def test_genomic_controls_match_chromosome_gc_and_length_without_replacement():
    sequences=pd.DataFrame(dict(sequence=["ACGT"*10,"ACGT"*10,"ACGT"*10,"ACGT"*10],chrom=["chr1","chr2","chr1","chr3"]))
    tokens=pd.DataFrame(dict(sequence_index=[0,1,2,3],is_support=[True,True,False,False]))
    pairs=match_controls(sequences,tokens)
    assert len(pairs)==1
    assert pairs[0]["present_index"]==0 and pairs[0]["absent_index"]==2


def test_sequence_pair_null_does_not_manufacture_signal_from_zero_differences():
    effect,interval,p,adjusted=paired_inference(np.zeros((4,2)),repetitions=99)
    assert np.all(effect==0) and np.all(interval==0)
    assert np.all(p==1) and np.all(adjusted==1)
