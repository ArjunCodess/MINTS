"""Cache-independent byte and semantic audit of the mapped-query study."""
from pathlib import Path, PureWindowsPath
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import math
import numpy as np
import pandas as pd
from src.assay_stats import genomic_clusters
from src.utils import sha256_file
from src.variant_statistics import cluster_summary
ROOT = Path(__file__).resolve().parents[1]

def audit(root=ROOT, raw=False, raw_root=None):
    diagnostic = root / 'results/control_diagnostics'
    study = root / 'results/mapped_variant'
    for directory in (diagnostic, study):
        receipt = json.loads((directory / 'execution.json').read_text())
        if not (receipt['status'] == 'completed' and (not receipt['source_changed_during_run'])):
            raise ValueError('Mapped integrity check failed at original line 22')
        for name, digest in receipt['artifacts'].items():
            if not sha256_file(directory / name) == digest:
                raise ValueError((directory, name))
        if not sha256_file(directory / 'protocol.json') == receipt['protocol_sha256']:
            raise ValueError('Mapped integrity check failed at original line 25')
        if directory == study:
            for name, digest in receipt['source_sha256'].items():
                if not sha256_file(study / 'source_snapshot' / name) == digest:
                    raise ValueError(name)
    source = pd.read_csv(root / 'results/adastra_exploratory/candidates.csv.gz')
    from src.adastra_candidates import select_candidates
    expected = select_candidates(source, 12288)
    membership = pd.read_csv(study / 'membership.csv')
    if not (len(source) == 512556 and len(membership) == 12288):
        raise ValueError('Mapped integrity check failed at original line 33')
    pd.testing.assert_frame_equal(expected, membership.drop(columns='cohort'))
    diag = pd.read_csv(diagnostic / 'cases.csv')
    preds = pd.read_csv(diagnostic / 'candidate_predicates.csv.gz')
    eligible = pd.read_csv(study / 'eligibility.csv')
    if not membership.variant_id.tolist() == diag.variant_id.tolist() == eligible.variant_id.tolist():
        raise ValueError('Mapped integrity check failed at original line 38')
    summary = json.loads((study / 'summary.json').read_text())
    if not summary['reference_verified'] == int(eligible.reference_verified.sum()) == 12288:
        raise ValueError('Mapped integrity check failed at original line 40')
    if not summary['motif_eligible'] == int(eligible.motif_eligible.sum()) == 433:
        raise ValueError('Mapped integrity check failed at original line 41')
    if not summary['correspondence_eligible'] == int(eligible.correspondence_eligible.sum()) == 433:
        raise ValueError('Mapped integrity check failed at original line 42')
    if not summary['matched_controls'] == int(eligible.matched_control.sum()) == 61:
        raise ValueError('Mapped integrity check failed at original line 43')
    if not diag.loc[diag.cohort == 'original4096', 'original'].sum() == 0:
        raise ValueError('Mapped integrity check failed at original line 44')
    if not diag.loc[diag.cohort == 'expansion', 'original'].sum() == 1:
        raise ValueError('Mapped integrity check failed at original line 45')
    for cohort, group in diag.groupby('cohort'):
        saved = json.loads((diagnostic / 'summary.json').read_text())['cohorts'][cohort]
        if not saved['records'] == len(group):
            raise ValueError('Mapped integrity check failed at original line 48')
        for key, value in saved.items():
            if key != 'records':
                if not int(group[key].sum()) == value:
                    raise ValueError((cohort, key))
    for key in ('original', 'local_only', 'local_no_width', 'local_geometry32', 'local_geometry64'):
        retained = set(preds.loc[preds[key], 'variant_id'])
        if not retained == set(diag.loc[diag[key], 'variant_id']):
            raise ValueError(key)
    cases = json.loads((study / 'cases.json').read_text())
    scores = json.loads((study / 'scores.json').read_text())
    if not [c['variant_id'] for c in cases] == [s['variant_id'] for s in scores]:
        raise ValueError('Mapped integrity check failed at original line 56')
    if not len(cases) == 61:
        raise ValueError('Mapped integrity check failed at original line 57')
    frame = pd.read_csv(study / 'case_scores.csv')
    for c, s in zip(cases, scores, strict=True):
        clean, alt, sham = (c[k] for k in ('reference_sequence', 'alternate_sequence', 'sham_sequence'))
        i, j = (c['variant_index'], c['sham_index'])
        if not (i == 102 and len(clean) == len(alt) == len(sham) == 204):
            raise ValueError('Mapped integrity check failed at original line 62')
        if not c['position_1based'] - 1 - c['window_start_0based'] == i:
            raise ValueError('Mapped integrity check failed at original line 63')
        if not c['window_end_0based'] - c['window_start_0based'] == 204:
            raise ValueError('Mapped integrity check failed at original line 64')
        if not [k for k, (a, b) in enumerate(zip(clean, alt)) if a != b] == [i]:
            raise ValueError('Mapped integrity check failed at original line 65')
        if not [k for k, (a, b) in enumerate(zip(clean, sham)) if a != b] == [j]:
            raise ValueError('Mapped integrity check failed at original line 66')
        if not (clean[i] == clean[j] == c['reference'] and alt[i] == sham[j] == c['alternate']):
            raise ValueError('Mapped integrity check failed at original line 67')
        if not (clean[i - 1:i + 2] == clean[j - 1:j + 2] and abs(i - j) <= 16):
            raise ValueError('Mapped integrity check failed at original line 68')
        frozen = preds[(preds.variant_id == c['variant_id']) & preds.local_no_width].sort_values(['distance_bp', 'sham_index'])
        if not int(frozen.iloc[0].sham_index) == j:
            raise ValueError('Mapped integrity check failed at original line 70')
        if not c['queries'] == [r['query'] for r in s['query_diagnostics']]:
            raise ValueError('Mapped integrity check failed at original line 71')
        for q, r in zip(c['queries'], s['query_diagnostics'], strict=True):
            a, b = q['span']
            if not (b - a == q['width'] and b > a):
                raise ValueError('Mapped integrity check failed at original line 74')
            if not (b <= min(i, j) or a > max(i, j)):
                raise ValueError('Mapped integrity check failed at original line 75')
            if not (len(q['indices']) == 3 and len(set((x[a:b] for x in (clean, alt, sham)))) == 1):
                raise ValueError('Mapped integrity check failed at original line 76')
            if not not (a < c['motif_span'][1] and b > c['motif_span'][0]):
                raise ValueError('Mapped integrity check failed at original line 77')
            masked = []
            for ids, k in zip(c['ids'], q['indices'], strict=True):
                if not ids[k] == q['token_id']:
                    raise ValueError('Mapped integrity check failed at original line 80')
                value = ids.copy()
                value[k] = -1
                masked.append(tuple(value))
            if not len(set(masked)) == 3:
                raise ValueError('Mapped integrity check failed at original line 83')
            if not max(r['identity_errors'] + r['final_restoration_errors'] + r['repeat_errors']) <= 0.0002:
                raise ValueError('Mapped integrity check failed at original line 84')
            if not r['normalization_error'] <= 1e-12:
                raise ValueError('Mapped integrity check failed at original line 85')
            if not all((math.isfinite(r[k]) and 0 <= r[k] <= math.log(2) + 1e-12 for k in ('variant_js', 'sham_js'))):
                raise ValueError('Mapped integrity check failed at original line 86')
        weights = [q['width'] for q in c['queries']]
        v = float(np.average([r['variant_js'] for r in s['query_diagnostics']], weights=weights))
        h = float(np.average([r['sham_js'] for r in s['query_diagnostics']], weights=weights))
        if not (abs(v - s['variant_divergence']) < 1e-12 and abs(h - s['sham_divergence']) < 1e-12):
            raise ValueError('Mapped integrity check failed at original line 90')
        if not (abs(v - h - s['contrast']) < 1e-12 and s['implementation_valid']):
            raise ValueError('Mapped integrity check failed at original line 91')
        row = frame[frame.variant_id == c['variant_id']].iloc[0]
        if not abs(row.contrast - s['contrast']) < 1e-12:
            raise ValueError('Mapped integrity check failed at original line 93')
        if raw:
            relative = PureWindowsPath(s['raw_logits_path'])
            path = Path(raw_root) / relative.name if raw_root else root.joinpath(*relative.parts)
            if not sha256_file(path) == s['raw_logits_sha256']:
                raise ValueError('Mapped integrity check failed at original line 96')
            import torch
            from src.variant_assay import js_divergence
            values = np.load(path)
            if not np.array_equal(values['query_indices'], [q['indices'] for q in c['queries']]):
                raise ValueError('Raw query index correspondence differs')
            if not np.array_equal(values['query_spans'], [q['span'] for q in c['queries']]):
                raise ValueError('Raw nucleotide spans differ')
            if not np.array_equal(values['weights'], [q['width'] for q in c['queries']]):
                raise ValueError('Raw query weights differ')
            if not np.isfinite(values['logits']).all():
                raise ValueError('Mapped integrity check failed at original line 100')
            for logits, r in zip(values['logits'], s['query_diagnostics'], strict=True):
                if not abs(js_divergence(torch.tensor(logits[0]), torch.tensor(logits[1])) - r['variant_js']) < 1e-12:
                    raise ValueError('Mapped integrity check failed at original line 102')
                if not abs(js_divergence(torch.tensor(logits[0]), torch.tensor(logits[2])) - r['sham_js']) < 1e-12:
                    raise ValueError('Mapped integrity check failed at original line 103')
    names = np.array([[f"{c['chrom']}:{c['window_start_0based']}-{c['window_end_0based']}"] * 3 for c in cases])
    sequences = np.array([[c[k] for k in ('reference_sequence', 'alternate_sequence', 'sham_sequence')] for c in cases])
    clusters = genomic_clusters(names, block_bp=1000000, sequences=sequences)
    if not clusters.tolist() == [c['cluster'] for c in cases] == frame.cluster.tolist():
        raise ValueError('Mapped integrity check failed at original line 107')
    primary = cluster_summary(frame.contrast, frame.cluster, repetitions=10000)
    for k in primary:
        if not (primary[k] == summary['primary'][k] or abs(primary[k] - summary['primary'][k]) < 1e-12):
            raise ValueError('Mapped integrity check failed at original line 109')
    if not summary['genomic_proxy_clusters'] == len(set(clusters)) == 60:
        raise ValueError('Mapped integrity check failed at original line 110')
    if not summary['implementation_valid'] == 61:
        raise ValueError('Mapped integrity check failed at original line 111')
    if not summary['full_boundary_stable_cases'] == int(frame.full_boundary_stable.sum()) == 15:
        raise ValueError('Mapped integrity check failed at original line 112')
    if not summary['full_boundary_stable_clusters'] == frame[frame.full_boundary_stable].cluster.nunique() == 15:
        raise ValueError('Mapped integrity check failed at original line 113')
    if not summary['width_matched_cases'] == int(frame.edit_width_matched.sum()) == 11:
        raise ValueError('Mapped integrity check failed at original line 114')
    if not summary['query_index_shift_cases'] == int(frame.query_index_shift.sum()) == 29:
        raise ValueError('Mapped integrity check failed at original line 115')
    width = frame[frame.edit_width_matched]
    width_summary = cluster_summary(width.contrast, width.cluster, repetitions=10000)
    for k in width_summary:
        if not abs(width_summary[k] - summary['width_matched'][k]) < 1e-12:
            raise ValueError('Mapped integrity check failed at original line 118')
    if not (summary['gates']['sequence_feasibility'] and summary['gates']['implementation_valid']):
        raise ValueError('Mapped integrity check failed at original line 119')
    if not (summary['gates']['tokenization_isolation'] and (not summary['gates']['biological_qc'])):
        raise ValueError('Mapped integrity check failed at original line 120')
    if not summary['gates']['native_sensitivity'] == (primary['ci_low'] > 0):
        raise ValueError('Mapped integrity check failed at original line 121')
    if not (not summary['gates']['head_search'] and (not summary['gates']['confirmation'])):
        raise ValueError('Mapped integrity check failed at original line 122')
    if not (summary['biological_qc_cases'] is None and summary['independent_donors'] is None):
        raise ValueError('Mapped integrity check failed at original line 123')
    ledger = json.loads((study / 'inspection_ledger.json').read_text())
    if not (ledger['native_outcomes_opened'] and (not ledger['confirmation_assigned'])):
        raise ValueError('Mapped integrity check failed at original line 125')
    if not ledger['membership_sha256'] == sha256_file(study / 'membership.csv'):
        raise ValueError('Mapped integrity check failed at original line 126')
    return dict(status='verified', selected=12288, scored=61, clusters=60, raw_logits_checked=raw, native_gate='failed', head_search='not run', confirmation='not run')
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', action='store_true')
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--raw-root', type=Path)
    a = p.parse_args()
    print(json.dumps(audit(a.root, a.raw, a.raw_root), indent=2))
