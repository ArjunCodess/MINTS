"""Read-only reanalysis of historical outputs without overwriting their provenance."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .assay_alignment import exact_offsets
from .assay_stats import genomic_clusters, paired_auc_difference, absolute_patching_summary
from .patching import stream_sparse_patch_positions
from .counterfactuals import char_span_to_token_span
from .utils import write_json, sha256_file


def run_diagnostics(tokenizer,config,output,bootstrap_samples=2000):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    root=config.paths.project_root
    pair_path=root/'results/cross_model/review_tata_heldout/counterfactuals/promoter_tata_batch_activation_patching_pairs.tsv'
    effects_path=root/'results/cross_model/review_tata_heldout/patching/promoter_tata_batch_dnabert_activation_patching_pair_effects.npz'
    pairs=pd.read_csv(pair_path,sep='\t');alignment=[]
    for row in pairs.itertuples():
        clean=exact_offsets(tokenizer,row.clean_sequence)
        corrupt=exact_offsets(tokenizer,row.corrupted_sequence)
        span=char_span_to_token_span(row.clean_sequence,row.start,row.end,tokenizer)
        positions=stream_sparse_patch_positions(len(clean),span,max_positions=max(8,4*int(np.ceil(np.log2(len(clean))))))
        mismatch=[i for i in positions if i>=len(corrupt) or clean[i]!=corrupt[i]]
        alignment.append(dict(sequence_id=row.sequence_id,same_token_count=len(clean)==len(corrupt),
            mismatched_patched_positions=len(mismatch),mismatched_indices=json.dumps(mismatch),
            clean_offsets=json.dumps(clean),corrupted_offsets=json.dumps(corrupt)))
    pd.DataFrame(alignment).to_csv(output/'historical_alignment.csv',index=False)
    with np.load(effects_path) as payload:
        scores={k:payload[k].copy() for k in payload.files}
    absolute=absolute_patching_summary(scores['clean_scores'],scores['corrupted_scores'],scores['patched_scores'],
                                      pairs.clean_sequence.to_numpy(),bootstrap_samples,config.data.seed)
    rows=[]
    for i,layer in enumerate(scores['layers']):
        for head in range(scores['patched_scores'].shape[-1]):
            j=i*scores['patched_scores'].shape[-1]+head
            pm=scores['restoration'][:,i,head]
            loo=(pm.sum()-pm)/(len(pm)-1)
            rows.append(dict(layer=int(layer),head=head,mean_pm=float(pm.mean()),median_pm=float(np.median(pm)),
                pm_loo_low=float(loo.min()),pm_loo_high=float(loo.max()),
                **{key:float(absolute[key][j]) for key in ['mean','median','ci_low','ci_high','loo_low','loo_high']}))
    pd.DataFrame(rows).to_csv(output/'historical_absolute_effects.csv',index=False)
    comparison=[]
    sources=[pair_path,effects_path]
    for task in config.data.task_names:
        path=root/f'results/review/{task}_predictions.csv';sources.append(path)
        predictions=pd.read_csv(path)
        clusters=genomic_clusters(predictions.name.to_numpy())
        for unit,codes in [('sequence',None),('genomic_block',clusters)]:
            result=paired_auc_difference(predictions.label,predictions.probe_probability,predictions.kmer_probability,
                                        codes,bootstrap_samples,config.data.seed)
            comparison.append(dict(task=task,resampling_unit=unit,first='probe',second='kmer',**result))
    pd.DataFrame(comparison).to_csv(output/'historical_paired_comparisons.csv',index=False)
    write_json(output/'diagnostics_manifest.json',dict(
        inputs={str(p.relative_to(root)):sha256_file(p) for p in sources},
        pairs=len(pairs),pairs_with_misaligned_patch_positions=sum(r['mismatched_patched_positions']>0 for r in alignment),
        negative_denominators=absolute['negative_corruption_pairs'],
        interpretation='historical patching is retained for audit only; new controlled, aligned studies are separate',
        inference='fixed classifiers; paired sequence and 1Mb genomic-block intervals',
        bootstrap_samples=bootstrap_samples,seed=config.data.seed))
    return comparison
