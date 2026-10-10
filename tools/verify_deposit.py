"""Verify the sequence-free deposit from saved numbers, without model downloads."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from src.assay_stats import paired_head_inference, paired_auc_difference
from src.probing import bootstrap_probe_confidence_intervals
from src.variant_statistics import cluster_summary
from src.intervention_study import secondary_corrections
from src.inference import holm_adjust


def verify(root):
    # A GitHub checkout does not need the separate local software ZIP.
    software = {'files': {}}
    if (root / 'code_manifest.json').exists():
        software = json.loads((root / 'code_manifest.json').read_text())
        for name, entry in software['files'].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'Software checksum mismatch: {name}')
    manifest = json.loads((root / 'deposit_manifest.json').read_text())
    for name, entry in manifest['files'].items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {name}')
    expected = json.loads((root / 'numerical/headlines.json').read_text())['estimates']
    estimates = {}

    def compare(result, saved, keys):
        for key in keys:
            if not np.isclose(result[key], saved[key], atol=1e-12, rtol=0):
                raise ValueError(f'Estimate mismatch for {key}: {result[key]} != {saved[key]}')

    clusters = pd.read_csv(root / 'paper/figures/intervention_sequence_clusters.csv')
    for motif, folder in [('CTCF', 'ctcf/interventions'), ('TATA', 'patching/promoter_tata')]:
        prefix = root / 'results/hardened' / folder
        effects = pd.read_csv(prefix / 'confirmation_pair_effects.csv')
        settings = json.loads((prefix / 'study_manifest.json').read_text())
        group = effects[(effects.scheme == 'edit') & (effects.direction == 'denoise')]
        pivot = group.pivot(index=['sequence_id', 'candidate_index'], columns='role', values='absolute_effect')
        means = (pivot.motif - pivot.sham).groupby(level='sequence_id').mean()
        membership = clusters[clusters.motif == motif].set_index('sequence_id').loc[means.index]
        if not np.allclose(means, membership.contrast, atol=1e-12, rtol=0):
            raise ValueError('Saved cluster membership does not align with recomputed effects')
        result = paired_head_inference(means.to_numpy(), membership.cluster.to_numpy(),
                                      settings['permutations'], settings['bootstrap_samples'], settings['seed'])
        estimates[motif] = {k: float(result[k][0]) for k in ('mean', 'ci_low', 'ci_high', 'p')}
        compare(estimates[motif], expected[motif], estimates[motif])
        revised = secondary_corrections(pd.read_csv(prefix / 'confirmation_summary.csv'))
        saved = pd.read_csv(root / f'paper/figures/{motif.lower()}_confirmation_summary.csv')
        for column in ('secondary_sequence_holm_p', 'secondary_block_holm_p'):
            if not np.allclose(revised[column], saved[column], equal_nan=True, atol=1e-12, rtol=0):
                raise ValueError('Secondary Holm correction mismatch')
    sensitivity = pd.read_csv(root / 'results/hardened/genomic/native_sensitivity_inference.csv')
    if not np.allclose(holm_adjust(sensitivity.p), sensitivity.sensitivity_family_holm_p, atol=1e-12, rtol=0):
        raise ValueError('Overall sensitivity Holm correction mismatch')
    counts = pd.read_csv(root / 'paper/figures/sensitivity_counts.csv')
    for row in counts.itertuples():
        group = sensitivity[(sensitivity.fraction == row.fraction) & (sensitivity.geometry == row.geometry)
            & (sensitivity.metric == 'base_density') & (sensitivity.resampling_unit == 'genomic_block')]
        if len(group) != 144 or int(((group['mean'] > 0) & (group.max_stat_p < .05)).sum()) != row.within_configuration or int(((group['mean'] > 0) & (group.sensitivity_family_holm_p < .05)).sum()) != row.overall:
            raise ValueError('Sensitivity family count mismatch')
    native = pd.read_csv(root / 'numerical/native_pilot.csv')
    settings = json.loads((root / 'results/native_endpoint/protocol.json').read_text())
    result = paired_head_inference((native.motif_loss - native.sham_loss).to_numpy(), native.cluster.to_numpy(),
                                  settings['permutations'], settings['bootstrap_samples'], settings['seed'])
    estimates['native_pilot'] = {k: float(result[k][0]) for k in ('mean', 'ci_low', 'ci_high', 'p')}
    compare(estimates['native_pilot'], expected['native_pilot'], estimates['native_pilot'])
    prediction = pd.read_csv(root / 'numerical/prediction.csv')
    settings = json.loads((root / 'results/hardened/protocol.json').read_text())
    aucs = {}
    for method, saved in expected['prediction']['estimates'].items():
        ci = bootstrap_probe_confidence_intervals(prediction.label.to_numpy(), prediction[method].to_numpy(),
             prediction[method].to_numpy() >= .5, settings['seed'], settings['bootstrap_samples'])['auroc']
        aucs[method] = dict(auroc=roc_auc_score(prediction.label, prediction[method]), ci_low=ci[0], ci_high=ci[1])
        compare(aucs[method], saved, aucs[method])
    delta = paired_auc_difference(prediction.label, prediction.baseline_plus_residual,
        prediction.motif_position_composition, prediction.cluster, settings['bootstrap_samples'], settings['seed'])
    compare(delta, expected['prediction']['paired_difference'], ['difference', 'ci_low', 'ci_high', 'clusters'])
    estimates['prediction'] = dict(estimates=aucs, paired_difference=delta)
    mapped = pd.read_csv(root / 'results/mapped_variant/case_scores.csv')
    raw = json.loads((root / 'numerical/raw_manifest.json').read_text())

    def js(a, b):
        a = a.astype('float64'); b = b.astype('float64')
        a -= logsumexp(a); b -= logsumexp(b)
        middle = np.logaddexp(a, b) - np.log(2)
        return max(0., float(.5 * (np.sum(np.exp(a) * (a - middle)) + np.sum(np.exp(b) * (b - middle)))))

    for item in raw:
        with np.load(root / item['file'], allow_pickle=False) as values:
            if set(values.files) != {'logits', 'query_indices', 'query_spans', 'weights'}:
                raise ValueError('Unexpected raw-array contents')
            if any(values[k].dtype.kind not in 'fiu' or not np.isfinite(values[k]).all() for k in values.files):
                raise ValueError('Nonfinite or nonnumeric raw arrays')
            logits, weights = values['logits'], values['weights']
            variant = np.average([js(x[0], x[1]) for x in logits], weights=weights)
            sham = np.average([js(x[0], x[2]) for x in logits], weights=weights)
        row = mapped[mapped.variant_id == item['variant_id']].iloc[0]
        compare(dict(v=variant, s=sham, d=variant-sham),
                dict(v=row.variant_divergence, s=row.sham_divergence, d=row.contrast), ['v', 's', 'd'])
    estimates['mapped'] = cluster_summary(mapped.contrast, mapped.cluster, repetitions=10000)
    compare(estimates['mapped'], expected['mapped'], estimates['mapped'])
    summary = json.loads((root / 'results/mapped_variant/summary.json').read_text())
    if summary['gates']['native_sensitivity'] or summary['gates']['head_search'] or summary['gates']['confirmation']:
        raise ValueError('Failed gates changed')
    return dict(status='verified', files=len(manifest['files']), software_files=len(software['files']), raw_cases=len(raw),
        secondary_holm='verified', overall_sensitivity_holm='verified',
        estimates=estimates, scope='Numerical reproduction using saved clusters; excludes sequence reconstruction and new model inference')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.data_root)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
