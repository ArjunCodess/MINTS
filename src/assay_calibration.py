"""Known-mechanism attention fixtures; calibration does not validate biology."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from .assay_alignment import base_attention_density
from .controlled_edits import MotifDefinition
from .utils import write_json


@dataclass
class KnownAttentionFixture:
    """Explicit QK/value/output computations with an oracle motif feature.

    The implanted feature is deliberately known. This checks assay sensitivity
    to computation placement and strength, not whether a real encoder learns it.
    """
    kind: str
    strength: float
    seed: int

    def forward(self, sequence, motif):
        count=len(sequence)
        support=np.zeros(count)
        for start,end,_ in motif.hits(sequence):
            support[start:end]=1.
        gc=np.asarray([base in 'GC' for base in sequence],dtype=float)
        base=np.asarray(['ACGT'.index(b) for b in sequence])
        rng=np.random.default_rng(self.seed)
        heads=4
        q=np.ones((heads,count,1))
        k=np.zeros_like(q)
        v=np.zeros_like(q)
        if self.kind in {'single','query_specific','distributed'}:
            k[0,:,0]=self.strength*support
            v[0,:,0]=support
            if self.kind=='query_specific':
                q[0,:,0]=0.
                q[0,-1,0]=1.
            if self.kind=='distributed':
                k[1,:,0]=self.strength*support
                v[0,:,0]=support/2
                v[1,:,0]=support/2
        elif self.kind=='composition':
            k[0,:,0]=self.strength*gc
            v[0,:,0]=gc
        elif self.kind=='randomized':
            q[:,:,0]=rng.normal(size=(heads,4))[:,base]
            k[:,:,0]=rng.normal(size=(heads,4))[:,base]
            v[:,:,0]=rng.normal(size=(heads,4))[:,base]
        else:
            raise ValueError(f"Unknown fixture kind: {self.kind}")
        logits=np.einsum('hid,hjd->hij',q,k)
        attention=softmax(logits,axis=-1)
        context=attention@v
        # The query-specific fixture writes to its final query; other readouts pool.
        readout=context[:,-1,0].sum() if self.kind=='query_specific' else context[:,:,0].mean(axis=1).sum()
        return dict(attention=attention,content_qk=logits,context=context,score=float(readout),support=support)


def _dna_pair(motif_sequence, position, instances, seed, length=128):
    rng=np.random.default_rng(seed)
    sequence=''.join(rng.choice(list('ACGT'),size=length))
    starts=[position]
    if instances>1:
        starts.append(min(length-len(motif_sequence),position+40))
    for start in starts:
        sequence=sequence[:start]+motif_sequence+sequence[start+len(motif_sequence):]
    motif=MotifDefinition('synthetic',pattern=motif_sequence)
    hits=motif.hits(sequence)
    if not hits:
        raise ValueError("Synthetic motif implantation failed")
    corrupted=list(sequence)
    for start,end,_ in hits:
        replacement=''.join(rng.permutation(list(sequence[start:end])))
        for _ in range(100):
            if replacement!=sequence[start:end] and replacement!=motif_sequence:
                break
            replacement=''.join(rng.permutation(list(sequence[start:end])))
        corrupted[start:end]=replacement
    corrupted=''.join(corrupted)
    # Remove chance motif occurrences by choosing another deterministic permutation.
    if motif.hits(corrupted):
        return _dna_pair(motif_sequence,position,instances,seed+10_000,length)
    return sequence,corrupted,motif,hits


def run_calibration(output, repetitions=24, seed=1729):
    if repetitions<4:
        raise ValueError("Calibration needs independent discovery and confirmation repetitions")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for motif_index,motif_sequence in enumerate(['TATAAA','CCGCGAGGAGGCAG']):
        for kind in ['single','query_specific','distributed','composition','randomized']:
            for strength in [.25,1.,4.]:
                for position in [8,48,88]:
                    for instances in [1,2]:
                        for repeat in range(repetitions):
                            fixture_seed=seed+repeat+100*motif_index
                            clean,corrupt,motif,hits=_dna_pair(motif_sequence,position,instances,fixture_seed)
                            fixture=KnownAttentionFixture(kind,strength,fixture_seed)
                            before,after=fixture.forward(clean,motif),fixture.forward(corrupt,motif)
                            span=hits[0][:2]
                            offsets=[(i,i+1) for i in range(len(clean))]
                            density=base_attention_density(before['attention'],offsets,span)
                            local_density=base_attention_density(before['attention'],offsets,span,(span[0],span[1]))
                            support=before['support']; scores=before['content_qk'].mean(axis=1)
                            r=np.array([np.corrcoef(s,support)[0,1] if np.std(s)>0 else 0. for s in scores])
                            # Replace the real pre-output head context, then evaluate the same native readout.
                            effects=[];edit_effects=[]
                            for head in range(4):
                                full=after['context'].copy();full[head]=before['context'][head]
                                edit=after['context'].copy()
                                positions=np.flatnonzero(support)
                                edit[head,positions]=before['context'][head,positions]
                                score=lambda c:float(c[:,-1,0].sum() if kind=='query_specific' else c[:,:,0].mean(axis=1).sum())
                                effects.append(score(full)-after['score'])
                                edit_effects.append(score(edit)-after['score'])
                            best=int(np.argmax(effects))
                            rows.append(dict(motif=motif_sequence,kind=kind,strength=strength,position=position,instances=instances,
                                repeat=repeat,partition='discovery' if repeat<repetitions//2 else 'confirmation',
                                known_detector=kind in {'single','query_specific','distributed'},
                                qk_r=float(r[best]),density=float(density[best]),local_query_density=float(local_density[best]),
                                absolute_effect=float(effects[best]),edit_only_effect=float(edit_effects[best]),
                                total_clean_minus_corrupted=before['score']-after['score']))
    table=pd.DataFrame(rows)
    discovery=table[table.partition=='discovery']
    choices=[]
    for r_cut in [.1,.3,.5]:
        for density_cut in [1.,1.25,2.]:
            for effect_cut in [.001,.01,.05]:
                passing=(discovery.qk_r>=r_cut)&(discovery.density>=density_cut)&(discovery.absolute_effect>=effect_cut)
                false_positive=float(passing[~discovery.known_detector].mean())
                sensitivity=float(passing[discovery.known_detector].mean())
                if false_positive<=.05:
                    choices.append((sensitivity,-false_positive,-effect_cut,r_cut,density_cut,effect_cut))
    if not choices:
        raise RuntimeError("No calibration threshold meets the discovery false-positive constraint")
    _,_,_,r_cut,density_cut,effect_cut=max(choices)
    table['passes_calibrated']=(table.qk_r>=r_cut)&(table.density>=density_cut)&(table.absolute_effect>=effect_cut)
    table['passes_historical']=(table.qk_r>=.5)&(table.density>=2.)
    table.to_csv(output/'calibration_observations.csv',index=False)
    summaries=[]
    for keys,group in table[table.partition=='confirmation'].groupby(['motif','kind','strength','position','instances']):
        summaries.append(dict(zip(['motif','kind','strength','position','instances'],keys),
                         observations=len(group),calibrated_detection_rate=float(group.passes_calibrated.mean()),
                         historical_detection_rate=float(group.passes_historical.mean()),
                         mean_absolute_effect=float(group.absolute_effect.mean()),mean_edit_only_effect=float(group.edit_only_effect.mean())))
    pd.DataFrame(summaries).to_csv(output/'calibration_summary.csv',index=False)
    confirmation=table[table.partition=='confirmation']
    manifest=dict(seed=seed,repetitions=repetitions,thresholds=dict(qk_r=r_cut,density=density_cut,absolute_effect=effect_cut),
                  selection='thresholds selected on discovery fixture seeds with false-positive rate <= .05',
                  confirmation_sensitivity=float(confirmation.loc[confirmation.known_detector,'passes_calibrated'].mean()),
                  confirmation_false_positive_rate=float(confirmation.loc[~confirmation.known_detector,'passes_calibrated'].mean()),
                  detector_feature='oracle motif feature explicitly implanted in QK and V; native fixture output is known',
                  scope='synthetic attention computations with base tokens; does not establish sensitivity in pretrained DNABERT',
                  limitations=['finite fixtures, strengths, and background distributions','no learned detector calibration','no biological validation'])
    write_json(output/'calibration_manifest.json',manifest)
    return manifest
