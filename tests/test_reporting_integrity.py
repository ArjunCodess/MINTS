import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.intervention_study import intervention_clusters, secondary_corrections
from src.assay_stats import paired_head_inference
from src.inference import holm_adjust
from tools.build_hardened_artifacts import sensitivity_counts

ROOT = Path(__file__).resolve().parents[1]


def test_repeated_cluster_observations_use_block_holm():
    values = np.array([1.] * 20 + [-1.] * 2 + [.2] * 3)
    clusters = np.array([0] * 20 + [1] * 2 + [2] * 3)
    sequence = paired_head_inference(values, repetitions=199, bootstrap_samples=100)['p'][0]
    block = paired_head_inference(values, clusters, repetitions=199, bootstrap_samples=100)['p'][0]
    report = secondary_corrections(pd.DataFrame(dict(analysis_scope=['primary', 'secondary exploratory', 'secondary exploratory'], sequence_p=[.01, sequence, .4], block_p=[.01, block, .6])))
    assert sequence != block
    assert np.allclose(report.secondary_block_holm_p.iloc[1:], holm_adjust([block, .6]))
    assert not np.array_equal(report.secondary_sequence_holm_p.iloc[1:], report.secondary_block_holm_p.iloc[1:])
    assert pd.isna(report.secondary_block_holm_p.iloc[0])


def test_edited_identity_and_rc_join_distant_loci_transitively():
    names = ['chr1:1-10', 'chr2:2000000-2000010', 'chr3:5000000-5000010']
    def pair(i, clean, edit, sham):
        return dict(motif=dict(sequence_id=names[i], clean_sequence=clean, corrupted_sequence=edit), sham=dict(corrupted_sequence=sham))
    pairs = [pair(0, 'AACCCC', 'ACTGAA', 'AAATTT'), pair(1, 'CCGTAA', 'TTCAGT', 'CCCCAA'),
             pair(1, 'CCGTAA', 'GGTACC', 'AGAGAA'), pair(2, 'GGGTTT', 'GGTACC', 'ATTATA')]
    result = intervention_clusters(names, pairs)
    assert len(set(result)) == 1


@pytest.mark.artifact
def test_saved_sensitivity_correction_families():
    source = ROOT/'results/hardened/genomic'
    native = pd.read_csv(source/'native_sensitivity_inference.csv')
    runs = json.loads((source/'native_sensitivity_manifest.json').read_text())['runs']
    counts = sensitivity_counts(native, runs)
    assert counts.overall.tolist() == [0]*6
    assert counts.within_configuration.tolist() == [0, 1, 7, 44, 12, 55]


@pytest.mark.artifact
@pytest.mark.parametrize('name', ['membership.csv', 'cases.json'])
def test_corrupt_mapped_membership_or_query_is_rejected(tmp_path, name):
    import shutil
    from tools.audit_mapped_variant import audit
    for folder in ('mapped_variant', 'control_diagnostics'):
        shutil.copytree(ROOT/'results'/folder, tmp_path/'results'/folder)
    path = tmp_path/'results/mapped_variant'/name
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):
        audit(tmp_path)


@pytest.mark.artifact
def test_corrupt_logits_rejected_with_relocated_raw_package(tmp_path):
    from tools.audit_mapped_variant import audit
    scores = json.loads((ROOT/'results/mapped_variant/scores.json').read_text())
    from pathlib import PureWindowsPath
    (tmp_path/PureWindowsPath(scores[0]['raw_logits_path']).name).write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        audit(ROOT, raw=True, raw_root=tmp_path)
