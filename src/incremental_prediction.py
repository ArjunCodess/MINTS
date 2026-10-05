"""Validation-selected frozen readouts and same-population incremental baselines."""
from __future__ import annotations

from itertools import product
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, average_precision_score

from .assay_stats import genomic_clusters, paired_auc_difference
from .integrity import COORD
from .probing import _gc_content_features, bootstrap_probe_confidence_intervals
from .utils import write_json


def baseline_features(sequences, names, motif=None):
    kmers = ["".join(k) for k in product("ACGT", repeat=2)]
    rows = []
    for sequence, name in zip(sequences, names):
        text = str(sequence).upper()
        match = COORD.search(str(name))
        if not match:
            raise ValueError(f"Missing genomic coordinate in predictive baseline: {name}")
        hits = [] if motif is None else motif.hits(text)
        rows.append([text.count(base) / max(1, len(text)) for base in "ACGT"] +
                    [sum(text[i:i+2] == k for i in range(len(text)-1)) / max(1,len(text)-1) for k in kmers] +
                    [len(text), len(hits), max((h[2] for h in hits), default=0.),
                     np.mean([(a+b)/2/len(text) for a,b,_ in hits]) if hits else 0.,
                     int(match[2])/250_000_000, (int(match[3])-int(match[2]))/1000])
    return np.asarray(rows, dtype=float)


def select_regularization(train_x, train_y, validation_x, validation_y, seed=1729, grid=(.01,.1,1.)):
    if np.unique(train_y).size != 2 or np.unique(validation_y).size != 2:
        raise ValueError("Training and validation populations must contain both labels")
    candidates = []
    best = None
    for c in grid:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c,max_iter=3000,class_weight="balanced",random_state=seed))
        model.fit(train_x, train_y)
        score = float(roc_auc_score(validation_y,model.predict_proba(validation_x)[:,1]))
        candidates.append(dict(c=float(c),validation_auroc=score))
        if best is None or score > best[0]:
            best = (score, model, float(c))
    return best[1], best[2], candidates


def classifier_digest(model):
    scaler, classifier = model.steps[0][1], model.steps[-1][1]
    digest = hashlib.sha256()
    for values in [scaler.mean_,scaler.scale_,classifier.coef_,classifier.intercept_,classifier.classes_]:
        digest.update(np.asarray(values).tobytes())
    return digest.hexdigest()


def compare_readouts(task, training, validation, testing, output, motif=None, seed=1729, bootstrap_samples=2000):
    """Fit only on train, choose C on validation, keep all models fixed on test."""
    from pathlib import Path
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    train_x, train_y, train_names, train_sequences = training
    val_x, val_y, val_names, val_sequences = validation
    test_x, test_y, test_names, test_sequences = testing
    populations=[(train_names,train_sequences),(val_names,val_sequences),(test_names,test_sequences)]
    chromosomes=[{COORD.search(str(n))[1] for n in names} for names,_ in populations]
    if any(chromosomes[i]&chromosomes[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError("Train, validation and test chromosomes must be disjoint")
    from .integrity import sequence_hash
    identities=[{sequence_hash(s) for s in seq} for _,seq in populations]
    if any(identities[i]&identities[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError("Exact or reverse-complement inputs cross predictive partitions")
    base_train=baseline_features(train_sequences,train_names,motif)
    base_val=baseline_features(val_sequences,val_names,motif)
    base_test=baseline_features(test_sequences,test_names,motif)
    definitions={"gc":(_gc_content_features(train_sequences),_gc_content_features(val_sequences),_gc_content_features(test_sequences)),
                 "motif_position_composition":(base_train,base_val,base_test),
                 "residual":(train_x,val_x,test_x),
                 "baseline_plus_residual":(np.column_stack([base_train,train_x]),np.column_stack([base_val,val_x]),np.column_stack([base_test,test_x]))}
    predictions=pd.DataFrame(dict(name=test_names,label=test_y))
    classifiers, selection, metrics = {}, {}, []
    for method,(tx,vx,ex) in definitions.items():
        model,c,history=select_regularization(tx,train_y,vx,val_y,seed)
        probability=model.predict_proba(ex)[:,1]
        predictions[method]=probability
        ci=bootstrap_probe_confidence_intervals(test_y,probability,probability>=.5,seed=seed,n_bootstraps=bootstrap_samples)
        metrics.append(dict(task=task,method=method,auroc=float(roc_auc_score(test_y,probability)),
                            auprc=float(average_precision_score(test_y,probability)),
                            auroc_ci_low=ci['auroc'][0],auroc_ci_high=ci['auroc'][1],c=c,test_examples=len(test_y)))
        classifiers[method]=model
        selection[method]=dict(c=c,validation_scores=history,classifier_sha256=classifier_digest(model))
    # Sequence baseline is tuned on the same validation population, with training-only vocabulary.
    kmer=make_pipeline(TfidfVectorizer(analyzer="char",ngram_range=(3,6),lowercase=False,min_df=2,max_features=50000,dtype=np.float64),
                       LogisticRegression(max_iter=2000,class_weight="balanced",random_state=seed,solver="liblinear"))
    best=None; history=[]
    for c in [.01,.1,1.]:
        kmer.set_params(logisticregression__C=c)
        kmer.fit(list(train_sequences),train_y)
        score=float(roc_auc_score(val_y,kmer.predict_proba(list(val_sequences))[:,1]))
        history.append(dict(c=c,validation_auroc=score))
        if best is None or score>best[0]:
            import copy
            best=(score,copy.deepcopy(kmer),c)
    kmer=best[1]
    predictions['kmer']=kmer.predict_proba(list(test_sequences))[:,1]
    ci=bootstrap_probe_confidence_intervals(test_y,predictions.kmer.to_numpy(),predictions.kmer.to_numpy()>=.5,seed=seed,n_bootstraps=bootstrap_samples)
    metrics.append(dict(task=task,method="kmer",auroc=float(roc_auc_score(test_y,predictions.kmer)),
                        auprc=float(average_precision_score(test_y,predictions.kmer)),auroc_ci_low=ci['auroc'][0],auroc_ci_high=ci['auroc'][1],c=best[2],test_examples=len(test_y)))
    selection['kmer']=dict(c=best[2],validation_scores=history)
    predictions.to_csv(output/f'{task}_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(output/f'{task}_metrics.csv',index=False)
    clusters=genomic_clusters(test_names,np.asarray(test_sequences)[:,None])
    comparisons=[]
    for a,b in [("residual","kmer"),("baseline_plus_residual","motif_position_composition"),("residual","gc")]:
        for unit,codes in [("sequence",None),("genomic_block",clusters)]:
            comparison=paired_auc_difference(test_y,predictions[a],predictions[b],clusters=codes,repetitions=bootstrap_samples,seed=seed)
            comparisons.append(dict(task=task,first=a,second=b,resampling_unit=unit,**comparison))
    pd.DataFrame(comparisons).to_csv(output/f'{task}_paired_comparisons.csv',index=False)
    receipt=dict(task=task,seed=seed,selection_partition="validation only; chromosome 18/19",
                 classifier_selection=selection,train_examples=len(train_y),validation_examples=len(val_y),test_examples=len(test_y),
                 partitions=[sorted(c) for c in chromosomes],
                 membership_sha256=[hashlib.sha256(json.dumps(sorted(map(str,names))).encode()).hexdigest() for names,_ in populations],
                 target="trained frozen residual readout decision function; not pretrained biological output",
                 validation_auroc=max(r['validation_auroc'] for r in selection['residual']['validation_scores']),
                 independent_test_auroc=next(m['auroc'] for m in metrics if m['method']=='residual'),
                 inference_scope="fixed classifiers; sequence and 1Mb genomic-block bootstrap; no checkpoint variability")
    write_json(output/f'{task}_selection.json',receipt)
    return classifiers['residual'],receipt
