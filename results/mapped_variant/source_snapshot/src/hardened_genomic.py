"""Recomputed motif definitions, token-balanced controls, and native assays."""
from pathlib import Path
import numpy as np
import pandas as pd
from .assay_stats import genomic_clusters, paired_head_inference
from .motif_scoring import load_jaspar_ctcf_motif, motif_pssm, score_sequence_tokens, default_support_threshold
from .native_assay import capture_localization
from .utils import write_json, sha256_file, progress


def motif_token_frame(sequences, tokenizer, motif, threshold, aggregation):
    rows=[]
    for index,row in enumerate(sequences.itertuples()):
        scores=score_sequence_tokens(str(row.sequence),tokenizer,motif,index,str(row.name),threshold,aggregation)
        support=set(scores.support_tokens)
        for token,(start,end) in enumerate(scores.token_offsets):
            rows.append(dict(sequence_index=index,sequence_id=row.name,token=token,char_start=start,char_end=end,
                             motif_score=float(scores.token_scores[token]),is_support=token in support))
    return pd.DataFrame(rows)


def inference_tables(differences,names,components,permutations,bootstrap_samples,seed,sequences=None):
    codes=genomic_clusters(names,sequences)
    rows=[]
    for unit,clusters in [('sequence_pair',None),('genomic_block',codes)]:
        if clusters is not None and np.unique(clusters).size<2:
            continue
        stats=paired_head_inference(differences,clusters,permutations,bootstrap_samples,seed)
        for index,(layer,head) in enumerate(components):
            rows.append(dict(layer=layer,head=head,resampling_unit=unit,clusters=stats['clusters'],
                             **{key:float(value[index]) for key,value in stats.items() if isinstance(value,np.ndarray)}))
    return rows


def run_genomic_sensitivity(bundle,config,output,limit=None,pair_limit=None,fractions=(.7,.8,.9),
                            aggregations=('max','mean','overlap_weighted'),bootstrap_samples=1000,permutations=9999,seed=1729):
    """Each support fraction reruns scoring and matching. Alternate PWM token
    aggregation changes QK inference; support always follows nucleotide hits.
    """
    from tools.review_ctcf_controls import match_controls
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=config.paths.ctcf_dir/'ctcf_gm12878_sequences.tsv'
    all_sequences=pd.read_csv(path,sep='\t')
    sequences=all_sequences.copy()
    if limit is not None and limit<len(sequences):
        sequences=sequences.groupby('chrom',group_keys=False).sample(
            frac=min(1.,limit/len(sequences)),random_state=seed).sort_index().reset_index(drop=True)
    motif=load_jaspar_ctcf_motif(config=config)
    pssm=motif_pssm(motif)
    all_summary=[];manifests=[]
    for fraction in fractions:
        threshold=default_support_threshold(pssm,fraction)
        tokens=motif_token_frame(sequences,bundle.tokenizer,motif,threshold,'max')
        grouped={int(i):group for i,group in tokens.groupby('sequence_index')}
        for geometry in ['nucleotide_matched','token_balanced']:
            pairs=match_controls(sequences,tokens,strict_geometry=geometry=='token_balanced',seed=seed)
            eligible=len(pairs)
            if pair_limit is not None and len(pairs)>pair_limit:
                keep=np.random.default_rng(seed).choice(len(pairs),pair_limit,replace=False)
                pairs=[pairs[i] for i in sorted(keep)]
            stem=f'f{fraction:.2f}_{geometry}'
            pd.DataFrame(pairs).to_csv(output/f'{stem}_pairs.csv',index=False)
            if len(pairs)<2:
                manifests.append(dict(fraction=fraction,geometry=geometry,status='insufficient_matched_pairs',eligible_pairs=eligible))
                continue
            values={};names=[];sequence_pairs=[];max_error=0.;diagnostics=[]
            for index,pair in enumerate(pairs):
                present=grouped[pair['present_index']]
                supported=present[present.is_support]
                run=supported.token.to_numpy()
                if np.any(np.diff(run)>1):
                    run=run[:np.flatnonzero(np.diff(run)>1)[0]+1]
                first=supported[supported.token.isin(run)]
                start,end=int(first.char_start.min()),int(first.char_end.max())
                roles=[];pair_names=[];pair_sequences=[]
                for role in ['present','absent']:
                    row=sequences.iloc[pair[f'{role}_index']]
                    sequence=str(row.sequence)
                    pair_sequences.append(sequence)
                    a=start if role=='present' else round(start*len(sequence)/pair['present_length'])
                    b=end if role=='present' else min(len(sequence),a+end-start)
                    scores={aggregation:score_sequence_tokens(sequence,bundle.tokenizer,motif,
                            threshold=threshold,aggregation=aggregation) for aggregation in aggregations}
                    capture=capture_localization(bundle,sequence,(a,b),scores[aggregations[0]].token_scores)
                    max_error=max(max_error,capture['max_context_error'])
                    for aggregation in aggregations:
                        if aggregation==aggregations[0]:
                            capture[f'qk_{aggregation}']=capture['qk_r']
                        else:
                            other=capture_localization(bundle,sequence,(a,b),scores[aggregation].token_scores)
                            capture[f'qk_{aggregation}']=other['qk_r']
                            max_error=max(max_error,other['max_context_error'])
                    capture.pop('qk_r',None)
                    roles.append(capture)
                    pair_names.append(f'{row.chrom}:{row.start}-{row.end}')
                    diagnostics.append(dict(pair=index,role=role,sequence_id=pair_names[-1],real_tokens=capture['real_tokens'],
                        target_tokens=capture['target_tokens'],target_start=a,target_end=b,
                        nucleotide_length=len(sequence),gc=pair[f'{role}_gc']))
                names.append(pair_names)
                sequence_pairs.append(pair_sequences)
                for key in roles[0]:
                    if isinstance(roles[0][key],np.ndarray):
                        values.setdefault(key,[]).append([roles[0][key],roles[1][key]])
                if index%100==0:
                    progress(f'native sensitivity {stem}: {index+1}/{len(pairs)} pairs')
            arrays={k:np.asarray(v) for k,v in values.items()}
            arrays['sequence_names']=np.asarray(names)
            np.savez_compressed(output/f'{stem}_scores.npz',**arrays)
            pd.DataFrame(diagnostics).to_csv(output/f'{stem}_balance.csv',index=False)
            first=next(v for k,v in arrays.items() if k!='sequence_names')
            layers,heads=first.shape[-2:]
            components=[(l,h) for l in range(layers) for h in range(heads)]
            for metric,values in arrays.items():
                if metric=='sequence_names':
                    continue
                valid=np.isfinite(values).all(axis=(1,2,3))
                if valid.sum()<2:
                    continue
                differences=(values[valid,0]-values[valid,1]).reshape(valid.sum(),-1)
                inference=inference_tables(differences,np.asarray(names)[valid],components,
                                           permutations,bootstrap_samples,seed,np.asarray(sequence_pairs)[valid])
                chromosome_rows=[]
                chromosomes=np.asarray([name[0].split(":")[0] for name in np.asarray(names)[valid]])
                for chrom in np.unique(chromosomes):
                    group=differences[chromosomes==chrom]
                    for j,(l,h) in enumerate(components):
                        chromosome_rows.append(dict(chromosome=chrom,layer=l,head=h,pairs=len(group),mean=float(group[:,j].mean())))
                pd.DataFrame(chromosome_rows).to_csv(output/f"{stem}_{metric}_chromosomes.csv",index=False)
                for row in inference:
                    row.update(fraction=fraction,geometry=geometry,metric=metric,pairs=int(valid.sum()),
                               excluded_pairs=int((~valid).sum()))
                all_summary.extend(inference)
            manifests.append(dict(fraction=fraction,threshold=threshold,geometry=geometry,status='completed',
                                  eligible_pairs=eligible,evaluated_pairs=len(pairs),max_native_context_error=max_error))
    table=pd.DataFrame(all_summary)
    if not table.empty:
        from .inference import holm_adjust
        table['sensitivity_family_holm_p']=holm_adjust(table.p.to_numpy())
    table.to_csv(output/'native_sensitivity_inference.csv',index=False)
    write_json(output/'native_sensitivity_manifest.json',dict(seed=seed,source_sha256=sha256_file(path),
        source_sequences=len(all_sequences),evaluated_sequences=len(sequences),sequence_cap=limit,pair_cap=pair_limit,
        model_revision=config.model.revision,fractions=list(fractions),aggregations=list(aggregations),runs=manifests,
        query_assays=['all nucleotide queries, width weighted','queries within 30bp of target span'],
        attention_assays=['native ALiBi-inclusive','content-only softmax','position-only softmax'],
        inference='sequence-pair and transitive 1Mb genomic-block bootstrap and sign flips; Holm and simultaneous max statistics',
        multiplicity='144-head family within each assay; additional Holm over all reported sensitivity rows',
        claim_boundary='conditional association within CTCF peaks; no binding absence or causal native output'))
    return table
