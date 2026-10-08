"""Rebuild public paper figures and inserts from saved evidence, without model inference."""
from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils import sha256_file, write_json
from src.intervention_study import intervention_clusters, secondary_corrections
from src.variant_statistics import cluster_summary
from tools.audit_mapped_variant import audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/figures'
TEX = ROOT / 'paper/generated'
REVISION = 'cc09923652f0b361f73a36f382c4358dcf8b8100'


def table(headers,rows):
    return "\n".join([r"\begin{tabular}{"+"l"+"r"*(len(headers)-1)+"}",r"\toprule",
        " & ".join(headers)+r" \\",r"\midrule",
        *[" & ".join(map(str,row))+r" \\" for row in rows],r"\bottomrule",r"\end{tabular}"])

def sensitivity_counts(native, runs):
    rows = []
    for run in runs:
        t = native[(native.fraction == run['fraction']) &
                   (native.geometry == run['geometry']) &
                   (native.metric == 'base_density') &
                   (native.resampling_unit == 'genomic_block')]
        if len(t) != 144:
            raise ValueError('Expected the complete 144-head configuration')
        rows.append(dict(fraction=run['fraction'], geometry=run['geometry'],
                         eligible=run['eligible_pairs'], evaluated=run.get('evaluated_pairs', 0),
                         within_configuration=int(((t['mean'] > 0) & (t.max_stat_p < .05)).sum()),
                         overall=int(((t['mean'] > 0) & (t.sensitivity_family_holm_p < .05)).sum())))
    return pd.DataFrame(rows)



def build():
    OUT.mkdir(parents=True, exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)
    audit(ROOT)
    inputs = set()
    def read(name):
        path = ROOT / name
        inputs.add(path)
        return pd.read_csv(path)
    def load(name):
        path = ROOT / name
        inputs.add(path)
        return json.loads(path.read_text())
    certificate = load('results/correspondence_audit/query_certificate.json')
    for name, expected in certificate['generators'].items():
        if sha256_file(ROOT/name) != expected:
            raise ValueError('Correspondence diagnostic generator changed: '+name)
    for name, expected in certificate['inputs'].items():
        if name.startswith('results/') and sha256_file(ROOT/name) != expected:
            raise ValueError('Correspondence diagnostic input changed: '+name)
    if (certificate['cases'], certificate['clusters'], certificate['query_count']) != (61,60,438):
        raise ValueError('Saved correspondence population differs')
    studies, secondaries, memberships, diagnostics = [], [], [], []
    for folder, label in [('patching/promoter_tata', 'TATA'), ('ctcf/interventions', 'CTCF')]:
        prefix = 'results/hardened/' + folder + '/'
        old = read(prefix + 'confirmation_summary.csv')
        revised = secondary_corrections(old)
        revised.to_csv(OUT / (label.lower() + '_confirmation_summary.csv'), index=False)
        pairs_path = ROOT / (prefix + 'controlled_pairs.jsonl')
        inputs.add(pairs_path)
        pairs = [json.loads(line) for line in pairs_path.read_text().splitlines()]
        confirmation = [p for p in pairs if p['partition'] == 'confirmation']
        effects = read(prefix + 'confirmation_pair_effects.csv')
        primary = effects[(effects.scheme == 'edit') & (effects.direction == 'denoise')]
        pivot = primary.pivot(index=['sequence_id', 'candidate_index'], columns='role', values='absolute_effect')
        means = (pivot.motif - pivot.sham).groupby(level='sequence_id').mean()
        codes = intervention_clusters(means.index, confirmation)
        row = revised[revised.analysis_scope == 'primary'].iloc[0]
        if len(np.unique(codes)) != row.clusters or not np.isclose(means.mean(), row.mean_absolute_motif_minus_sham, atol=1e-12, rtol=0):
            raise ValueError('Identity clustering changes saved primary population or estimate')
        for name, mean, cluster in zip(means.index, means, codes, strict=True):
            memberships.append(dict(motif=label, sequence_id=name, contrast=mean, cluster=int(cluster)))
        studies.append([label, f'{int(row.layer)}/{int(row["head"])}', int(row.sequences), int(row.clusters),
                        f'{row.mean_absolute_motif_minus_sham:.4f}', f'[{row.block_ci_low:.4f}, {row.block_ci_high:.4f}]', f'{row.block_p:.4f}'])
        for r in revised[revised.analysis_scope != 'primary'].itertuples():
            secondaries.append([label, r.scheme.replace('_', r'\_'), r.direction,
                                f'{r.secondary_sequence_holm_p:.4f}', f'{r.secondary_block_holm_p:.4f}'])
        eligibility = load(prefix + 'eligibility.json')
        distances = [p['sham_distance_bp'] for p in confirmation]
        diagnostics.append(dict(motif=label, discovery_scanned=len(eligibility['discovery']),
            confirmation_scanned=len(eligibility['confirmation']), discovery_retained=eligibility['retained_discovery_sequences'],
            confirmation_retained=eligibility['retained_confirmation_sequences'], edits=len(confirmation),
            clusters=len(np.unique(codes)), sham_median_bp=float(np.median(distances)), sham_max_bp=max(distances)))
    pd.DataFrame(memberships).to_csv(OUT / 'intervention_sequence_clusters.csv', index=False)
    write_json(OUT / 'control_diagnostics.json', diagnostics)
    native = read('results/hardened/genomic/native_sensitivity_inference.csv')
    runs = load('results/hardened/genomic/native_sensitivity_manifest.json')['runs']
    counts = sensitivity_counts(native, runs)
    counts.to_csv(OUT / 'sensitivity_counts.csv', index=False)
    geom = [[f'{r.fraction:.2f}', r.geometry.replace('_', r'\_'), r.eligible, r.evaluated, r.within_configuration, r.overall] for r in counts.itertuples()]
    loc = read('results/hardened/ctcf/interventions/same_motif_localization.csv')
    loc = loc[loc.role == 'clean'].groupby('sequence_id')[['base_density', 'content_density', 'position_only_density']].mean().mean()
    row_means = read('results/hardened/ctcf/interventions/same_motif_localization.csv')
    row_means = row_means[row_means.role == 'clean'][['base_density', 'content_density', 'position_only_density']].mean()
    write_json(OUT / 'selected_head_localization.json', dict(sequence_weighted=loc.to_dict(), edit_row_weighted=row_means.to_dict()))
    metrics = read('results/hardened/ctcf/ctcf_accessible_peak_overlap_metrics.csv')
    comparisons = read('results/hardened/ctcf/ctcf_accessible_peak_overlap_paired_comparisons.csv')
    cmp = comparisons[(comparisons['first'] == 'baseline_plus_residual') & (comparisons.resampling_unit == 'genomic_block')].iloc[0]
    labels = {'motif_position_composition': 'Motif/position/composition', 'residual': 'Residual only', 'baseline_plus_residual': 'Baseline + residual'}
    preds = [[labels[r.method], f'{r.auroc:.4f}', f'[{r.auroc_ci_low:.4f}, {r.auroc_ci_high:.4f}]', r.test_examples] for r in metrics.itertuples() if r.method in labels]
    scores = read('results/mapped_variant/case_scores.csv')
    groups = read('results/mapped_variant/cluster_scores.csv')
    summary = load('results/mapped_variant/summary.json')
    primary = cluster_summary(scores.contrast, scores.cluster, repetitions=10000)
    if any(not np.isclose(primary[k], summary['primary'][k], atol=1e-12, rtol=0) for k in primary):
        raise ValueError('Mapped headline recomputation disagrees')
    variant, sham = float(groups.variant.mean()), float(groups.sham.mean())
    groups.to_csv(OUT / 'mapped_plot_data.csv', index=False)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.4, 2.8), layout='constrained')
    ax.scatter(groups.contrast, np.arange(len(groups)), s=14, color='#286e82')
    ax.axvline(0, color='black', linewidth=.8)
    ax.errorbar(primary['mean'], -6, xerr=[[primary['mean']-primary['ci_low']], [primary['ci_high']-primary['mean']]], fmt='D', color='#a34e24', capsize=4)
    ax.set(xlabel='Variant minus sham JS divergence (nats)', ylabel='Genomic proxy cluster', ylim=(-10, 62))
    ax.set_title('61 cases / 60 genomic proxy clusters', fontsize=11)
    # Presentation metadata must not make a saved-evidence rebuild time-dependent.
    fig.savefig(OUT / 'mapped_effect.pdf', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(OUT / 'mapped_effect.png', dpi=180); plt.close(fig)
    pilot = load('results/native_endpoint/summary.json')
    headline = dict(mapped=dict(cases=len(scores), variant_js=variant, sham_js=sham, **primary),
                    predictive=cmp.to_dict(), intervention=studies, pilot=pilot, localization=loc.to_dict(),
                    leave_cluster_out_range=[min(x['mean'] for x in summary['leave_cluster_out']), max(x['mean'] for x in summary['leave_cluster_out'])])
    write_json(OUT / 'headline_numbers.json', headline)
    numbers = f'''% Generated by tools/build_paper_artifacts.py from saved evidence.
\\newcommand{{\\MappedVariantJS}}{{{variant:.6f}}}
\\newcommand{{\\MappedShamJS}}{{{sham:.6f}}}
\\newcommand{{\\MappedMean}}{{{primary['mean']:.6f}}}
\\newcommand{{\\MappedLow}}{{{primary['ci_low']:.6f}}}
\\newcommand{{\\MappedHigh}}{{{primary['ci_high']:.6f}}}
\\newcommand{{\\IncrementalMean}}{{{cmp.difference:.4f}}}
\\newcommand{{\\IncrementalLow}}{{{cmp.ci_low:.4f}}}
\\newcommand{{\\IncrementalHigh}}{{{cmp.ci_high:.4f}}}
\\newcommand{{\\SelectedBase}}{{{loc.base_density:.4f}}}
\\newcommand{{\\SelectedContent}}{{{loc.content_density:.4f}}}
\\newcommand{{\\SelectedPosition}}{{{loc.position_only_density:.4f}}}
'''
    (TEX / 'numbers.tex').write_text(numbers, newline='\n')
    (TEX / 'interventions.tex').write_text(table(['Motif', 'Head', '$n$', 'Clusters', 'Effect', '95\\% block interval', '$p$'], studies), newline='\n')
    (TEX / 'prediction.tex').write_text(table(['Predictor', 'AUROC', '95\\% sequence interval', '$n$'], preds), newline='\n')
    (TEX / 'geometry.tex').write_text(table(['Fraction', 'Geometry', 'Eligible', 'Scored', 'Within', 'Overall'], geom), newline='\n')
    # Long secondary table is a separate supplement insert, using block Holm in its labeled column.
    (TEX / 'secondary.tex').write_text(table(['Motif', 'Scheme', 'Direction', 'Sequence Holm', 'Block Holm'], secondaries), newline='\n')
    outputs = [p for p in OUT.iterdir() if p.is_file() and p.name != 'lineage.json'] + list(TEX.glob('*.tex'))
    write_json(OUT / 'lineage.json', dict(revision='public-paper-reporting-v1', scientific_revision=REVISION,
        scope='Saved-evidence reporting revision; no model inference, membership, protocols or gates changed',
        inputs={p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(inputs)},
        generators={p.relative_to(ROOT).as_posix(): sha256_file(p) for p in [Path(__file__), ROOT/'src/intervention_study.py', ROOT/'tools/audit_mapped_variant.py', ROOT/'src/inference.py', ROOT/'src/assay_stats.py', ROOT/'src/variant_statistics.py']},
        outputs={p.relative_to(ROOT).as_posix(): sha256_file(p) for p in outputs}))
    print('Public paper artifacts rebuilt; mapped headlines recomputed; original execution receipts retained.')
    return OUT


if __name__ == '__main__':
    build()
