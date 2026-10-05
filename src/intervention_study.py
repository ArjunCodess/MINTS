"""Discovery-selected heads, independently evaluated controlled interventions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .assay_alignment import intervention_positions, validate_patch_alignment
from .assay_stats import genomic_clusters, paired_head_inference
from .controlled_edits import controlled_edit_pairs
from .incremental_prediction import classifier_digest
from .modeling import encoder_layers
from .patching import _score_sequence_with_probe, _cache_clean_attention_self_outputs, _patched_probe_score_with_positions
from .native_assay import capture_localization
from .utils import write_json, sha256_file, progress


def collect_controlled_pairs(table, motif, tokenizer, seed=1729, limit=32, edits_per_sequence=1, candidates=128):
    rows=table[table.label==1].copy()
    rows=rows.iloc[np.random.default_rng(seed).permutation(len(rows))]
    retained=[];audits=[]
    for i,row in enumerate(rows.itertuples()):
        pairs,audit=controlled_edit_pairs(str(row.sequence),str(row.name),motif,tokenizer,
                                         max_candidates=candidates,max_pairs=edits_per_sequence,seed=seed+i)
        audits.append(audit)
        retained.extend(pairs)
        if limit is not None and len({p['motif']['sequence_id'] for p in retained})>=limit:
            break
    return retained,audits


def evaluate_pair(bundle, pair, scorer, config, components, schemes=('edit',), directions=('denoise',), seed=1729):
    layers=tuple(sorted({layer for layer,_ in components}))
    clean=pair['motif']['clean_sequence']
    clean_score=_score_sequence_with_probe(bundle,clean,scorer,config)
    clean_cache=_cache_clean_attention_self_outputs(bundle,clean,layers)
    rows=[]
    for role in ['motif','sham']:
        record=pair[role]
        corrupt=record['corrupted_sequence']
        corrupt_score=_score_sequence_with_probe(bundle,corrupt,scorer,config)
        offsets=validate_patch_alignment(bundle.tokenizer,clean,corrupt)['clean_offsets']
        corrupt_cache=_cache_clean_attention_self_outputs(bundle,corrupt,layers) if 'reverse' in directions else None
        for scheme in schemes:
            positions=intervention_positions(offsets,(record['start'],record['end']),scheme,seed=seed)
            validate_patch_alignment(bundle.tokenizer,clean,corrupt,positions)
            for direction in directions:
                target_sequence,reference_cache,reference_score=(corrupt,clean_cache,corrupt_score) if direction=='denoise' else (clean,corrupt_cache,clean_score)
                for layer,head in components:
                    patched=_patched_probe_score_with_positions(bundle,target_sequence,scorer,config,reference_cache,layer,head,positions)
                    effect=patched-reference_score
                    rows.append(dict(sequence_id=record['sequence_id'],candidate_index=pair['candidate_index'],role=role,
                        scheme=scheme,direction=direction,layer=layer,head=head,clean_score=clean_score,
                        corrupted_score=corrupt_score,patched_score=patched,absolute_effect=effect,
                        clean_minus_corrupted=clean_score-corrupt_score,
                        edit_count=pair['edit_count'],sham_distance_bp=pair['sham_distance_bp'],
                        positions=json.dumps(positions),nucleotide_intervals=json.dumps([offsets[i] for i in positions])))
    return rows


def run_controlled_interventions(bundle, motif, scorer, discovery, confirmation, config, output,
                                 discovery_limit=12, confirmation_limit=64, candidates=128, bootstrap_samples=1000,
                                 permutations=9999, seed=1729, localization=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    from .integrity import sequence_hash
    shared_identities = {sequence_hash(s) for s in discovery.sequence} & {sequence_hash(s) for s in confirmation.sequence}
    if set(discovery.name)&set(confirmation.name) or shared_identities:
        raise ValueError('Discovery and confirmation inputs overlap')
    discovery_pairs,discovery_audit=collect_controlled_pairs(discovery,motif,bundle.tokenizer,seed,discovery_limit,1,candidates)
    confirmation_pairs,confirmation_audit=collect_controlled_pairs(confirmation,motif,bundle.tokenizer,seed+1,confirmation_limit,3,candidates)
    write_json(output/'eligibility.json',dict(discovery=discovery_audit,confirmation=confirmation_audit,
               candidate_selection='sequence/motif/edit properties only; no model-output filtering',
               planned_discovery_limit=discovery_limit,planned_confirmation_limit=confirmation_limit,
               retained_discovery_sequences=len(discovery_pairs),retained_confirmation_sequences=len({p['motif']['sequence_id'] for p in confirmation_pairs})))
    with (output/'controlled_pairs.jsonl').open('w',encoding='utf-8') as handle:
        for partition,pairs in [('discovery',discovery_pairs),('confirmation',confirmation_pairs)]:
            for pair in pairs:
                handle.write(json.dumps(dict(partition=partition,**pair))+'\n')
    if len(discovery_pairs)<2 or len({p['motif']['sequence_id'] for p in confirmation_pairs})<2:
        write_json(output/'study_manifest.json',dict(status='insufficient_eligible_population',motif=motif.name,
                   claim_boundary='no mechanistic discovery or confirmation claimed',eligibility='eligibility.json'))
        return None
    components=[(i,h) for i,layer in enumerate(encoder_layers(bundle.hf_model)) for h in range(layer.attention.self.num_attention_heads)]
    rows=[]
    for i,pair in enumerate(discovery_pairs):
        rows.extend(evaluate_pair(bundle,pair,scorer,config,components,seed=seed+i))
        progress(f'{motif.name}: discovery {i+1}/{len(discovery_pairs)}')
    table=pd.DataFrame(rows)
    table.to_csv(output/'discovery_pair_effects.csv',index=False)
    indexed=table.pivot(index=['sequence_id','candidate_index'],columns=['role','layer','head'],values='absolute_effect')
    effects=np.column_stack([(indexed[('motif',l,h)]-indexed[('sham',l,h)]).to_numpy() for l,h in components])
    selected=int(np.argmax(effects.mean(axis=0)))
    component=components[selected]
    frozen=dict(motif=motif.name,layer=component[0],head=component[1],
                criterion='maximum discovery mean motif-minus-transition-matched-sham absolute score effect',
                discovery_membership=list(indexed.index.get_level_values('sequence_id')),
                scorer_sha256=classifier_digest(scorer),primary_scheme='edit',primary_direction='denoise',
                primary_endpoint='motif_minus_sham_absolute_decision_score_change',
                selection_scope='discovery chromosomes only; independent confirmation',
                calibration_scope='synthetic fixture thresholds are not transferred to native biological claims')
    selection_path=output/'frozen_selection.json'
    write_json(selection_path,frozen)
    selection_hash=sha256_file(selection_path)
    # Discovery multiplicity remains explicit even though the confirmation head is frozen.
    names=indexed.index.get_level_values('sequence_id').to_numpy()
    clusters=genomic_clusters(names)
    if np.unique(clusters).size>=2:
        stats=paired_head_inference(effects,clusters,permutations,bootstrap_samples,seed)
        pd.DataFrame([dict(layer=l,head=h,**{k:float(v[j]) for k,v in stats.items() if isinstance(v,np.ndarray)})
                      for j,(l,h) in enumerate(components)]).to_csv(output/'discovery_inference.csv',index=False)
    rows=[];localization_rows=[]
    schemes=('edit','all','all_with_special','flank','random','sparse')
    for i,pair in enumerate(confirmation_pairs):
        rows.extend(evaluate_pair(bundle,pair,scorer,config,[component],schemes,('denoise','reverse'),seed+i))
        if localization:
            start,end=pair['motif']['start'],pair['motif']['end']
            for role,sequence in [('clean',pair['motif']['clean_sequence']),('motif',pair['motif']['corrupted_sequence']),('sham',pair['sham']['corrupted_sequence'])]:
                native=capture_localization(bundle,sequence,(start,end))
                l,h=component
                localization_rows.append(dict(sequence_id=pair['motif']['sequence_id'],candidate_index=pair['candidate_index'],role=role,
                    layer=l,head=h,**{k:float(native[k][l,h]) for k in ['base_density','local_query_density','content_density','position_only_density']},
                    max_context_error=native['max_context_error']))
        if i%10==0 or i+1==len(confirmation_pairs):
            progress(f'{motif.name}: confirmation {i+1}/{len(confirmation_pairs)}')
    confirm=pd.DataFrame(rows)
    confirm.to_csv(output/'confirmation_pair_effects.csv',index=False)
    summaries=[]
    for (scheme,direction),group in confirm.groupby(['scheme','direction']):
        pivot=group.pivot(index=['sequence_id','candidate_index'],columns='role',values='absolute_effect')
        # Candidate edits are averaged within sequence before inference to avoid pseudo-replication.
        per_sequence=(pivot.motif-pivot.sham).groupby(level='sequence_id').mean()
        values=per_sequence.to_numpy()
        codes=genomic_clusters(per_sequence.index.to_numpy())
        sequence_stats=paired_head_inference(values,None,permutations,bootstrap_samples,seed)
        stats=paired_head_inference(values,codes,permutations,bootstrap_samples,seed) if np.unique(codes).size>=2 else None
        loo=(values.sum()-values)/(len(values)-1)
        summaries.append(dict(scheme=scheme,direction=direction,layer=component[0],head=component[1],
            mean_absolute_motif_minus_sham=float(values.mean()),median=float(np.median(values)),
            loo_low=float(loo.min()),loo_high=float(loo.max()),sequence_ci_low=float(sequence_stats['ci_low'][0]),
            sequence_ci_high=float(sequence_stats['ci_high'][0]),sequence_p=float(sequence_stats['p'][0]),
            block_ci_low=float(stats['ci_low'][0]) if stats else None,block_ci_high=float(stats['ci_high'][0]) if stats else None,
            block_p=float(stats['p'][0]) if stats else None,clusters=int(np.unique(codes).size),sequences=len(values),
            analysis_scope='primary' if scheme=='edit' and direction=='denoise' else 'secondary exploratory'))
    summary=pd.DataFrame(summaries)
    # Secondary choices form a family; the primary fixed-head test remains distinct.
    from .inference import holm_adjust
    secondary=summary.analysis_scope=='secondary exploratory'
    summary['secondary_holm_p']=np.nan
    summary.loc[secondary,'secondary_holm_p']=holm_adjust(summary.loc[secondary,'sequence_p'])
    summary.to_csv(output/'confirmation_summary.csv',index=False)
    if localization_rows:
        pd.DataFrame(localization_rows).to_csv(output/'same_motif_localization.csv',index=False)
    if sha256_file(selection_path)!=selection_hash:
        raise RuntimeError('Frozen selection changed during confirmation')
    write_json(output/'study_manifest.json',dict(status='completed',motif=motif.name,seed=seed,
        frozen_selection_sha256=selection_hash,scorer_sha256=classifier_digest(scorer),
        discovery_sequences=len(discovery_pairs),confirmation_sequences=confirm.sequence_id.nunique(),
        confirmation_edits=len(confirmation_pairs),schemes=list(schemes),directions=['denoise','reverse'],
        primary_endpoint=frozen['primary_endpoint'],bootstrap_samples=bootstrap_samples,permutations=permutations,
        aggregation='average candidate edits within sequence, then genomic-block inference',
        resampling_assumptions='exchangeability for conditional observational genomic blocks; no biological randomization',
        claim_boundary='effects on a trained frozen readout; no native pretrained biological mechanism established'))
    return summary
