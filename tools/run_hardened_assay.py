"""Run isolated, audited mechanistic-assay hardening experiments.

All scientific outputs live in a separate tree. Historical artifacts are inputs,
never overwritten. Caps are explicit protocol fields, and insufficient retained
populations are recorded rather than promoted to successful mechanism evidence.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from dataclasses import replace
import gc
import importlib.metadata
import json
import platform
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from src.config import DEFAULT_CONFIG
from src.controlled_edits import MotifDefinition
from src.hardened_diagnostics import run_diagnostics
from src.assay_calibration import run_calibration
from src.incremental_prediction import compare_readouts
from src.intervention_study import run_controlled_interventions
from src.hardened_genomic import run_genomic_sensitivity
from src.ctcf_controls import download_accessibility,build_ctcf_case_control
from src.motif_scoring import load_jaspar_ctcf_motif,motif_pssm
from src.modeling import load_hooked_encoder,encode_sequences,forward_hidden_states
from src.activations import mean_pool_hidden,mean_pool_unpadded_hidden
from src.integrity import COORD,sequence_hash
from src.provenance import verified_import_manifest,verify_reproduced_caches
from src.utils import progress,write_json,sha256_file


STAGES=('diagnostics','calibration','prediction','patching','genomic','ctcf')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results/hardened'))
    p.add_argument('--stage',choices=('all',*STAGES),default='all')
    p.add_argument('--device',default='auto')
    p.add_argument('--seed',type=int,default=1729)
    p.add_argument('--bootstrap-samples',type=int,default=1000)
    p.add_argument('--permutations',type=int,default=9999)
    p.add_argument('--train-cap',type=int,default=None)
    p.add_argument('--discovery-pairs',type=int,default=12)
    p.add_argument('--confirmation-pairs',type=int,default=64)
    p.add_argument('--edit-candidates',type=int,default=128)
    p.add_argument('--genomic-sequences',type=int,default=None)
    p.add_argument('--genomic-pairs',type=int,default=256)
    p.add_argument('--ctcf-train-per-class',type=int,default=512)
    p.add_argument('--ctcf-validation-per-class',type=int,default=128)
    p.add_argument('--ctcf-test-per-class',type=int,default=128)
    p.add_argument('--calibration-repetitions',type=int,default=24)
    p.add_argument('--replace-stage',action='store_true',help='Explicitly rerun generated stages after implementation changes')
    return p


def load_populations(task,config,train_cap=None):
    paths=[config.paths.activations_dir/f'{task}_{split}_residual_mean.npz' for split in ['train','test']]
    comparison=config.paths.results_dir/'review/clean_feature_comparison.json'
    reproduction=config.paths.results_dir/'review/clean_reproduction.json'
    verify_reproduced_caches(paths,task,comparison,reproduction,config)
    arrays=[]
    for path in paths:
        with np.load(path) as saved:
            layer=list(saved['layers']).index(config.data.probe_layer)
            arrays.append((saved['residual_mean'][:,layer].copy(),saved['labels'].astype(int),
                           saved['names'].astype(str),saved['sequences'].astype(str)))
    training,testing=arrays
    chromosomes=np.array([COORD.search(n)[1].lower() for n in training[2]])
    validation=np.isin(chromosomes,['chr18','chr19'])
    if not validation.any():
        raise ValueError(f'{task} lacks independent validation chromosomes 18/19')
    def subset(data,indices):
        return tuple(a[indices] for a in data)
    train_indices=np.flatnonzero(~validation)
    if train_cap is not None and len(train_indices)>train_cap:
        from sklearn.model_selection import train_test_split
        train_indices,_=train_test_split(train_indices,train_size=train_cap,random_state=config.data.seed,
                                        stratify=training[1][train_indices])
    return subset(training,train_indices),subset(training,np.flatnonzero(validation)),testing,paths+[comparison,reproduction]


def table_from_population(population):
    _,y,names,sequences=population
    return pd.DataFrame(dict(name=names,label=y,sequence=sequences))


def fresh_residuals(bundle,table,config):
    import torch
    residual=[]
    for start in range(0,len(table),config.data.batch_size):
        seq=table.sequence.iloc[start:start+config.data.batch_size].astype(str).tolist()
        encoded=encode_sequences(bundle.tokenizer,seq,bundle.device)
        with torch.no_grad():
            hidden=forward_hidden_states(bundle.hf_model,encoded)[config.data.probe_layer]
            pooled=mean_pool_hidden(hidden,encoded['attention_mask']) if hidden.ndim==3 else mean_pool_unpadded_hidden(hidden,encoded['attention_mask'])
        residual.append(pooled.detach().cpu().numpy())
    return np.concatenate(residual)


def run(args):
    root=DEFAULT_CONFIG.paths.project_root
    output=(root/args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError('Hardened outputs must stay inside the project workspace')
    for key,value in vars(args).items():
        if isinstance(value,int) and not isinstance(value,bool) and value<1:
            raise ValueError(f'{key} must be positive')
    output.mkdir(parents=True,exist_ok=True)
    config=replace(DEFAULT_CONFIG,model=replace(DEFAULT_CONFIG.model,device=args.device,local_files_only=True),
                   data=replace(DEFAULT_CONFIG.data,seed=args.seed))
    protocol={k:v.as_posix() if isinstance(v,Path) else v for k,v in vars(args).items() if k not in {'replace_stage','stage','device'}}
    protocol.update(model_revision=config.model.revision,dataset_revision=config.data.hf_dataset_revision,
                    validation_chromosomes=['chr18','chr19'],test_chromosomes=['chr20','chr21'],
                    primary_endpoint='motif-minus-transition-matched-sham absolute probe decision-score change',
                    primary_scheme='edit-only nucleotide positions',primary_direction='denoise',
                    biological_claim='trained readout effects; not native pretrained binding causality',
                    ctcf_validation_gate='validation AUROC >= .6; tested before interventions',
                    prospective_status='protocol recorded before this execution; not an external preregistration')
    protocol_path=output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol and not args.replace_stage:
        raise ValueError('Protocol differs from this run; choose a fresh output directory')
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        archived_protocol=output/'history'/f'protocol_{sha256_file(protocol_path)}.json'
        archived_protocol.parent.mkdir(parents=True,exist_ok=True)
        if not archived_protocol.exists():
            archived_protocol.write_bytes(protocol_path.read_bytes())
    write_json(protocol_path,protocol)
    tokenizer=AutoTokenizer.from_pretrained(config.model.model_name,revision=config.model.revision,
                                            trust_remote_code=True,local_files_only=True)
    bundle=None;scorers={};populations={}
    def get_bundle():
        nonlocal bundle
        if bundle is None:
            bundle=load_hooked_encoder(config.model)
            progress(f'hardened assays: model loaded on {bundle.device}')
        return bundle
    def fit_task(task,motif):
        training,validation,testing,paths=load_populations(task,config,args.train_cap)
        verified_import_manifest(paths,output/'prediction'/f'{task}_cache_import.json',config,
            'Historical frozen caches are reused as measured inputs. Prior pinned clean reproduction matched their bytes; historical checkpoint identity is not independently inferred.')
        scorer,receipt=compare_readouts(task,training,validation,testing,output/'prediction',motif,args.seed,args.bootstrap_samples)
        scorers[task]=scorer
        populations[task]=(validation,testing)
        return scorer,receipt,validation,testing
    motifs=dict(promoter_tata=MotifDefinition('TATA',pattern='TATA[AT]A'),
                promoter_no_tata=MotifDefinition('TATA',pattern='TATA[AT]A'),
                splice_sites_donors=MotifDefinition('donor_GT',pattern='GT',scan_reverse=False,prefer_center=True),
                splice_sites_acceptors=MotifDefinition('acceptor_AG',pattern='AG',scan_reverse=False,prefer_center=True))
    stages=STAGES if args.stage=='all' else (args.stage,)
    overall=[]
    loaded_sources={p.relative_to(root).as_posix():sha256_file(p) for folder in ["src","tools"] for p in (root/folder).glob("*.py")}
    for stage in stages:
        destination=output/stage
        destination.mkdir(parents=True,exist_ok=True)
        status_path=destination/'execution.json'
        if status_path.exists() and args.replace_stage:
            history=output/'history'/stage
            history.mkdir(parents=True,exist_ok=True)
            archived=history/f'{sha256_file(status_path)}.json'
            if not archived.exists():
                archived.write_bytes(status_path.read_bytes())
        if status_path.exists() and not args.replace_stage:
            raise ValueError(f'Stage {stage} already has an execution record; preserve it or use explicit --replace-stage')
        start=time.perf_counter()
        environment=dict(python=platform.python_version(),command=[sys.executable,*sys.argv],
            protocol_sha256=sha256_file(protocol_path),stage=stage,status='running',
            packages={name:importlib.metadata.version(name) for name in ['numpy','torch','transformers','datasets','scikit-learn','scipy','pandas']},
            source_sha256=loaded_sources.copy())
        write_json(status_path,environment)
        progress(f'hardened stage: {stage}')
        try:
            with threadpool_limits(limits=1):
                if stage=='diagnostics':
                    run_diagnostics(tokenizer,config,destination,args.bootstrap_samples)
                elif stage=='calibration':
                    run_calibration(destination,args.calibration_repetitions,args.seed)
                elif stage=='prediction':
                    for task in config.data.task_names:
                        progress(f'validation-selected prediction: {task}')
                        fit_task(task,motifs[task])
                        if task not in {'promoter_tata','splice_sites_donors'}:
                            scorers.pop(task,None);populations.pop(task,None);gc.collect()
                elif stage=='patching':
                    for task in ['promoter_tata','splice_sites_donors']:
                        if task not in scorers:
                            scorer,_,validation,testing=fit_task(task,motifs[task])
                        else:
                            scorer=scorers[task];validation,testing=populations[task]
                        run_controlled_interventions(get_bundle(),motifs[task],scorer,
                            table_from_population(validation),table_from_population(testing),config,destination/task,
                            args.discovery_pairs,args.confirmation_pairs,args.edit_candidates,args.bootstrap_samples,args.permutations,args.seed,
                            localization=task=='promoter_tata')
                elif stage=='genomic':
                    run_genomic_sensitivity(get_bundle(),config,destination,args.genomic_sequences,args.genomic_pairs,
                                           bootstrap_samples=args.bootstrap_samples,permutations=args.permutations,seed=args.seed)
                elif stage=='ctcf':
                    accessibility=download_accessibility(output/'sources')
                    pssm=motif_pssm(load_jaspar_ctcf_motif(config=config))
                    motif=MotifDefinition('CTCF',pssm=pssm)
                    cohort=build_ctcf_case_control(config.paths.ctcf_dir/'ctcf_gm12878_sequences.tsv',accessibility,
                        config.paths.grch38_fasta,destination,motif,per_class_train=args.ctcf_train_per_class,
                        per_class_validation=args.ctcf_validation_per_class,per_class_test=args.ctcf_test_per_class,seed=args.seed)
                    x=fresh_residuals(get_bundle(),cohort,config)
                    np.savez_compressed(destination/'ctcf_fresh_residuals.npz',residual=x,names=cohort.name.to_numpy())
                    parts=[]
                    for partition in ['train','validation','test']:
                        indices=np.flatnonzero(cohort.partition.to_numpy()==partition)
                        parts.append((x[indices],cohort.label.to_numpy()[indices],cohort.name.to_numpy()[indices],cohort.sequence.to_numpy()[indices]))
                    scorer,receipt=compare_readouts('ctcf_accessible_peak_overlap',*parts,destination,motif,args.seed,args.bootstrap_samples)
                    validated=receipt['validation_auroc']>=.6
                    write_json(destination/'target_validation.json',dict(validation_gate=.6,validation_auroc=receipt['validation_auroc'],
                        status='predictive_readout_validated' if validated else 'predictive_readout_failed_gate',
                        output='trained frozen CTCF peak-overlap readout; does not represent pretrained binding output'))
                    if validated:
                        run_controlled_interventions(get_bundle(),motif,scorer,cohort[cohort.partition=='validation'],
                            cohort[cohort.partition=='test'],config,destination/'interventions',args.discovery_pairs,
                            args.confirmation_pairs,args.edit_candidates,args.bootstrap_samples,args.permutations,args.seed,localization=True)
                    else:
                        write_json(destination/'interventions_status.json',dict(status='not_run_target_failed_validation',
                            claim_boundary='no CTCF intervention or detector discovery claimed'))
            environment['status']='completed'
        except Exception as error:
            environment.update(status='failed',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            environment['seconds']=time.perf_counter()-start
            environment['artifacts']={p.relative_to(output).as_posix():sha256_file(p) for p in destination.rglob('*') if p.is_file() and p!=status_path}
            environment['source_changed_during_stage']=[name for name,digest in environment['source_sha256'].items() if sha256_file(root/name)!=digest]
            write_json(status_path,environment)
            overall.append(dict(stage=stage,status=environment['status'],seconds=environment['seconds'],execution_sha256=sha256_file(status_path)))
            consolidated=[]
            for name in STAGES:
                record=output/name/'execution.json'
                if record.exists():
                    saved=json.loads(record.read_text())
                    consolidated.append(dict(stage=name,status=saved['status'],seconds=saved['seconds'] if 'seconds' in saved else None,
                        execution_sha256=sha256_file(record),protocol_sha256=saved['protocol_sha256']))
            complete=len(consolidated)==len(STAGES) and all(r['status']=='completed' and r['protocol_sha256']==sha256_file(protocol_path) for r in consolidated)
            write_json(output/'run_manifest.json',dict(protocol_sha256=sha256_file(protocol_path),stages=consolidated,
                        status='completed' if complete else 'partial',requested_stages=list(stages)))
    return output


if __name__=='__main__':
    result=run(parser().parse_args())
    print(json.dumps(dict(output=str(result)),indent=2),flush=True)
