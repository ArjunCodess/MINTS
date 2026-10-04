"""Native-attention audit with matched motif-absent peak controls.

All available motif-absent peaks are eligible, with no debug cap. Match to
motif-present peaks without replacement on chromosome, GC within .02 and
length within 5%. This is a post-review observational design, not causal proof.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import time
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
from src.config import DEFAULT_CONFIG
from src.modeling import load_hf_components, encoder_layers, encode_sequences
from src.inference import holm_adjust
from src.integrity import COORD
from src.probing import _sequence_gc_fraction
from src.utils import sha256_file

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/review"
REVISION="7bce263b15377fc15361f52cfab88f8b586abda0"


def match_controls(sequences, tokens):
    support=tokens.groupby("sequence_index")["is_support"].any()
    if not np.array_equal(support.index.to_numpy(),np.arange(len(sequences))):
        raise ValueError("Motif scores must cover exactly the sequence table in row-index order")
    if "sequence_id" in tokens and "name" in sequences:
        grouped=tokens.groupby("sequence_index").sequence_id
        if not (grouped.nunique()==1).all() or not np.array_equal(
            grouped.first().astype(str).to_numpy(),sequences.name.astype(str).to_numpy()
        ):
            raise ValueError("Motif-score sequence IDs do not align with the sequence table")
    present=np.flatnonzero(support.to_numpy())
    absent=np.flatnonzero(~support.to_numpy())
    gc=np.asarray([_sequence_gc_fraction(s) for s in sequences.sequence])
    lengths=sequences.sequence.str.len().to_numpy()
    chromosomes=np.asarray([str(c) for c in sequences.chrom])
    rng=np.random.default_rng(1729)
    rng.shuffle(absent)
    available=set(present.tolist())
    pairs=[]
    for control in absent:
        candidates=np.asarray(sorted(available),dtype=int)
        eligible=(chromosomes[candidates]==chromosomes[control]) & (np.abs(gc[candidates]-gc[control])<=.02) & (np.abs(lengths[candidates]-lengths[control])<=.05*lengths[control])
        candidates=candidates[eligible]
        if not len(candidates):
            continue
        best=int(candidates[np.argmin(np.abs(gc[candidates]-gc[control]) + np.abs(lengths[candidates]-lengths[control])/lengths[control])])
        available.remove(best)
        pairs.append(dict(present_index=best,absent_index=int(control),present_gc=float(gc[best]),absent_gc=float(gc[control]),
                          present_length=int(lengths[best]),absent_length=int(lengths[control]),chromosome=chromosomes[control]))
    return pairs


def paired_inference(differences, repetitions=9999):
    """Sequence-pair sign-flip null; conditional exchangeability is assumed."""
    differences=np.asarray(differences,dtype=np.float64)
    observed=differences.mean(axis=0)
    rng=np.random.default_rng(1729)
    exceed=np.zeros(differences.shape[1],dtype=int)
    for start in range(0,repetitions,100):
        signs=rng.choice([-1.,1.],size=(min(100,repetitions-start),len(differences)))
        null=signs@differences/len(differences)
        exceed+=(np.abs(null)>=np.abs(observed)).sum(axis=0)
    p=(exceed+1)/(repetitions+1)
    draws=[]
    for _ in range(1000):
        indices=rng.integers(len(differences),size=len(differences))
        draws.append(differences[indices].mean(axis=0))
    ci=np.quantile(np.asarray(draws),[.025,.975],axis=0)
    return observed,ci,p,holm_adjust(p)


def sequence_qk_correlation(key_scores, motif_scores):
    """Within-sequence Pearson r; undefined sequences remain explicitly missing."""
    x=np.asarray(key_scores,dtype=float)
    y=np.asarray(motif_scores,dtype=float)
    finite=np.isfinite(y)&np.isfinite(x).all(axis=0)
    if finite.sum()<3:
        return np.full(x.shape[0],np.nan)
    x=x[:,finite]-x[:,finite].mean(axis=1,keepdims=True)
    y=y[finite]-y[finite].mean()
    denominator=np.sqrt((x*x).sum(axis=1)*(y*y).sum())
    return np.divide(x@y,denominator,out=np.full(x.shape[0],np.nan),where=denominator>0)


def main(config=DEFAULT_CONFIG):
    started=time.perf_counter()
    out=config.paths.results_dir/"review"
    out.mkdir(parents=True,exist_ok=True)
    sequence_path=config.paths.ctcf_dir/"ctcf_gm12878_sequences.tsv"
    token_path=config.paths.enrichment_dir/"ctcf_qk_alignment_token_motif_scores.csv"
    sequences=pd.read_csv(sequence_path,sep="\t")
    tokens=pd.read_csv(token_path)
    tokens["is_support"]=tokens.is_support.astype(str).str.lower().isin(["true","1"])
    if "chrom" not in sequences:
        if "chromosome" in sequences:
            sequences["chrom"]=sequences.chromosome
        else:
            sequences["chrom"]=[COORD.search(str(n))[1] for n in sequences.name]
    pairs=match_controls(sequences,tokens)
    pd.DataFrame(pairs).to_csv(out/"ctcf_genomic_control_pairs.csv",index=False)
    grouped={int(index):group for index,group in tokens.groupby("sequence_index",sort=False)}
    model_config=replace(config.model,revision=config.model.revision or REVISION)
    tokenizer,model,device=load_hf_components(model_config)
    model.eval()
    layers=encoder_layers(model)
    captured={}
    captured_qk={}
    errors=[]
    max_validation_error=0.
    handles=[]
    current_support=None
    current_real=None
    current_motif=None

    def make_hook(layer):
        def hook(module,inputs,output):
            nonlocal max_validation_error
            hidden,cu,max_length,indices,mask,bias=inputs
            count=hidden.shape[0]
            if int(cu.shape[0])!=2 or count!=int(max_length):
                raise ValueError("Native audit expects one unpadded sequence")
            qkv=module.Wqkv(hidden).reshape(count,3,module.num_attention_heads,module.attention_head_size)
            q=qkv[:,0].permute(1,0,2)
            k=qkv[:,1].permute(1,2,0)
            v=qkv[:,2].permute(1,0,2)
            content_logits=q@k/np.sqrt(module.attention_head_size)
            logits=content_logits+bias[0,:,:count,:count]
            attention=torch.softmax(logits,dim=-1)
            reconstructed=(attention@v).permute(1,0,2).reshape(count,-1)
            error=float((reconstructed-output).abs().max().item())
            max_validation_error=max(max_validation_error,error)
            if not torch.allclose(reconstructed,output,atol=2e-4,rtol=2e-4):
                raise ValueError(f"Native attention reconstruction failed: max error {error}")
            real=torch.as_tensor(current_real,device=device)
            target=torch.as_tensor(current_support,device=device)
            # Query mean on real nucleotide tokens. Multiply per-key mass by
            # number of real keys to express mass relative to uniform attention.
            score=attention[:,real][:,:,target].mean(dim=(1,2))*len(current_real)
            captured[layer]=score.detach().cpu().numpy()
            # Projection biases are included; ALiBi is excluded from content QK.
            key_scores=content_logits[:,real][:,:,real].mean(dim=1).detach().cpu().numpy()
            captured_qk[layer]=sequence_qk_correlation(key_scores,current_motif)
        return hook

    for layer,module in enumerate(layers):
        handles.append(module.attention.self.register_forward_hook(make_hook(layer)))
    values=[]
    qk_values=[]
    valid_pairs=[]
    try:
        with torch.no_grad():
            for index,pair in enumerate(pairs):
                present=grouped[pair["present_index"]]
                supported=present[present.is_support]
                start=int(supported.char_start.min())
                end=int(supported.char_end.max())
                # Use the first connected support run rather than filling gaps
                # between separate motif instances.
                starts=supported.token.to_numpy()
                run=starts[:np.flatnonzero(np.diff(starts)>1)[0]+1] if np.any(np.diff(starts)>1) else starts
                first=supported[supported.token.isin(run)]
                start,end=int(first.char_start.min()),int(first.char_end.max())
                scores=[]
                qk_scores=[]
                try:
                    for role in ["present","absent"]:
                        row_index=pair[f"{role}_index"]
                        sequence=str(sequences.iloc[row_index].sequence)
                        record=grouped[row_index]
                        encoded=encode_sequences(tokenizer,[sequence],device,max_length=None)
                        offsets=tokenizer(sequence,return_offsets_mapping=True,truncation=False)["offset_mapping"]
                        expected=list(zip(record.char_start.astype(int),record.char_end.astype(int)))
                        if offsets!=expected or len(offsets)!=encoded["input_ids"].shape[-1]:
                            raise ValueError("Tokenizer/alignment mismatch against saved motif scores")
                        span_start=start if role=="present" else round(start*len(sequence)/pair["present_length"])
                        span_end=end if role=="present" else min(len(sequence),span_start+(end-start))
                        current_real=[i for i,(a,b) in enumerate(offsets) if b>a]
                        current_motif=record.motif_score.to_numpy(dtype=float)[current_real]
                        current_support=[i for i,(a,b) in enumerate(offsets) if b>a and a<span_end and b>span_start]
                        if not current_support:
                            raise ValueError("Control span lacks token support")
                        captured.clear()
                        captured_qk.clear()
                        model(input_ids=encoded["input_ids"],attention_mask=encoded["attention_mask"],output_all_encoded_layers=False)
                        scores.append(np.stack([captured[layer] for layer in range(len(layers))]))
                        qk_scores.append(np.stack([captured_qk[layer] for layer in range(len(layers))]))
                    values.append(scores)
                    qk_values.append(qk_scores)
                    valid_pairs.append(pair)
                except ValueError as error:
                    errors.append(dict(pair=index,error=str(error)))
                if index%250==0:
                    print(f"{index+1}/{len(pairs)} pairs; valid={len(values)}",flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if len(values)<2:
        raise RuntimeError("Insufficient validated sequence pairs")
    values=np.asarray(values)
    qk_values=np.asarray(qk_values)
    np.savez_compressed(out/"ctcf_native_control_scores.npz",scores=values,qk_correlations=qk_values,
                        present_indices=np.asarray([p["present_index"] for p in valid_pairs]),
                        absent_indices=np.asarray([p["absent_index"] for p in valid_pairs]))
    observed,ci,p,adjusted=paired_inference((values[:,0]-values[:,1]).reshape(len(values),-1))
    heads=values.shape[-1]
    pd.DataFrame([dict(layer=i//heads,head=i%heads,mean_present_minus_absent=float(effect),
                       ci_low=float(ci[0,i]),ci_high=float(ci[1,i]),permutation_p=float(p[i]),holm_p=float(adjusted[i]),pairs=len(values))
                  for i,effect in enumerate(observed)]).to_csv(out/"ctcf_native_control_inference.csv",index=False)
    qk_valid=np.isfinite(qk_values).all(axis=(1,2,3))
    if qk_valid.sum()<2:
        raise RuntimeError("Insufficient pairs with defined sequence-level QK correlations")
    qk=qk_values[qk_valid]
    observed,ci,p,adjusted=paired_inference((qk[:,0]-qk[:,1]).reshape(len(qk),-1))
    present_mean=qk[:,0].mean(axis=0).reshape(-1)
    rng=np.random.default_rng(1729)
    draws=np.asarray([qk[rng.integers(len(qk),size=len(qk)),0].mean(axis=0).reshape(-1) for _ in range(1000)])
    present_ci=np.quantile(draws,[.025,.975],axis=0)
    pd.DataFrame([dict(layer=i//heads,head=i%heads,mean_present_r=float(present_mean[i]),
                       present_ci_low=float(present_ci[0,i]),present_ci_high=float(present_ci[1,i]),
                       mean_present_minus_absent=float(effect),ci_low=float(ci[0,i]),ci_high=float(ci[1,i]),
                       permutation_p=float(p[i]),holm_p=float(adjusted[i]),pairs=len(qk))
                  for i,effect in enumerate(observed)]).to_csv(out/"ctcf_sequence_qk_inference.csv",index=False)
    manifest=dict(command=[sys.executable,*sys.argv],entrypoint="tools.review_ctcf_controls.main(config)",
                  revision=model_config.revision,device=device,seed=1729,eligible_absent_peaks=int((~tokens.groupby("sequence_index").is_support.any()).sum()),
                  matched_pairs=len(pairs),valid_pairs=len(values),excluded_pairs=errors,max_native_context_error=max_validation_error,
                  query_positions="all nucleotide-spanning tokens, excludes specials",aggregation_unit="matched sequence pair",
                  score="native attention mean over real query positions and target keys, times number of real keys",
                  control="first connected motif support span projected to relative position in motif-absent peak",
                  matching="same chromosome; GC within .02; nucleotide length within 5%; without replacement",
                  inference="two-sided paired sign flips, 9999 permutations, Holm over 144 heads; 1000 pair-bootstrap marginal intervals",
                  qk_score="Pearson r within each sequence between continuous maximum-overlap PWM scores and scaled projection-bias-inclusive QK averaged over nucleotide queries; ALiBi excluded",
                  qk_valid_pairs=int(qk_valid.sum()),qk_undefined_pairs=int((~qk_valid).sum()),
                  qk_inference="present-minus-absent sequence-r contrasts; separate Holm family of 144 heads, 9999 sign flips; 1000 pair bootstraps; present means also have marginal intervals",
                  assumptions="observational matching; sequence-pair exchangeability conditional on matching; within-chromosome dependence not modeled",
                  claim_boundary="association of native attention with motif-present peak intervals, not motif detection or causal use",
                  seconds=time.perf_counter()-started,inputs={str(p):sha256_file(p) for p in [sequence_path,token_path]})
    (out/"ctcf_native_controls_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
