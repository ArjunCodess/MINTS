"""Exploratory native-endpoint diagnostics without changing the stopped pilot."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict

import numpy as np
import torch

from .assay_alignment import exact_offsets, validate_patch_alignment
from .assay_stats import genomic_clusters
from .controlled_edits import _record, edit_signature
from .native_endpoint import native_score, prepare_target


def target_geometry(span, target):
    a, b = span
    c, d = target
    if b <= c:
        return "upstream", c - b
    if a >= d:
        return "downstream", a - d
    return "overlap", 0


def match_target_geometry(motif_span, sham_span, target_span, tolerance_bp=2):
    """Same-side controls must not overlap, and must match target distance."""
    if tolerance_bp < 0:
        raise ValueError("Distance tolerance must be nonnegative")
    if motif_span[0] < sham_span[1] and sham_span[0] < motif_span[1]:
        return False
    side, gap = target_geometry(motif_span, target_span)
    sham_side, sham_gap = target_geometry(sham_span, target_span)
    return side == sham_side and side != "overlap" and abs(gap-sham_gap) <= tolerance_bp


def rematch_sham(tokenizer, pair, motif, protocol, tolerance_bp=2):
    """Hold the original motif edit and target fixed, search sequence-only controls.

    Targets and edits are not changed when a control is infeasible. A tight
    same-side distance match is structurally impossible for disjoint wide
    windows; this exclusion is returned before consulting model outputs.
    """
    target = prepare_target(tokenizer, pair, protocol)
    record = pair["motif"]
    clean = record["clean_sequence"]
    start, end = record["start"], record["end"]
    width = end-start
    if width > tolerance_bp:
        return None, dict(status="structurally_infeasible", width_bp=width,
                          minimum_same_side_gap_difference_bp=width, tolerance_bp=tolerance_bp)
    hits = {(a, b) for a, b, _ in motif.hits(clean)}
    offsets = exact_offsets(tokenizer, clean)
    signature = edit_signature(clean, record["corrupted_sequence"])
    motif_indices = [i for i, (a, b) in enumerate(offsets) if b>a and a<end and b>start]
    for a in range(len(clean)-width+1):
        b = a+width
        if not match_target_geometry((start, end), (a, b), target["span"], tolerance_bp):
            continue
        if any(a<y and b>x for x,y in hits):
            continue
        source = clean[a:b]
        letters, used = list(source), set()
        for (old, new), count in sorted(signature.items()):
            available = [i for i, base in enumerate(source) if base==old and i not in used]
            if len(available)<count:
                break
            for i in available[:count]:
                letters[i]=new
                used.add(i)
        else:
            changed = clean[:a]+"".join(letters)+clean[b:]
            if Counter(clean)!=Counter(changed) or {(x,y) for x,y,_ in motif.hits(changed)} != hits:
                continue
            try:
                alignment=validate_patch_alignment(tokenizer, clean, changed)
            except ValueError:
                continue
            indices=[i for i,(x,y) in enumerate(offsets) if y>x and x<b and y>a]
            if len(indices)!=len(motif_indices):
                continue
            if offsets[indices[-1]][1]-offsets[indices[0]][0] != offsets[motif_indices[-1]][1]-offsets[motif_indices[0]][0]:
                continue
            new_pair={**pair, "sham":asdict(_record(clean, changed, (a,b), record["sequence_id"], motif,
                                                  "target_geometry_matched_sham")), "sham_alignment":alignment}
            new_target=prepare_target(tokenizer,new_pair,protocol)
            if new_target["index"]==target["index"] and new_target["token_id"]==target["token_id"]:
                return dict(pair=new_pair,target=new_target), dict(status="retained", tolerance_bp=tolerance_bp)
    return None, dict(status="no_eligible_geometry_matched_sham", tolerance_bp=tolerance_bp)


def recovery_targets(tokenizer, table, chromosomes, cap, seed):
    """One random real token per fixed genomic window, with no outcome filtering."""
    rows=table[table.chrom.isin(chromosomes)].copy()
    rows["sequence_id"]=rows.chrom+":"+rows.start.astype(str)+"-"+rows.end.astype(str)
    rows=rows.sort_values("sequence_id").reset_index(drop=True)
    rng=np.random.default_rng(seed)
    chosen=rows.iloc[rng.permutation(len(rows))[:cap]]
    targets=[]
    for row in chosen.itertuples():
        if set(row.sequence)-set("ACGT"):
            raise ValueError("Calibration population must contain canonical DNA")
        offsets=exact_offsets(tokenizer,row.sequence)
        ids=list(tokenizer(row.sequence,add_special_tokens=True,truncation=False)["input_ids"])
        indices=[i for i,(a,b) in enumerate(offsets) if b>a and ids[i] not in tokenizer.all_special_ids]
        index=int(rng.choice(indices))
        a,b=offsets[index]
        target=dict(index=index,token_id=ids[index],span=[a,b])
        ids[index]=tokenizer.mask_token_id
        targets.append(dict(sequence_id=row.sequence_id,sequence=row.sequence,target=target,masked_ids=ids,
                            token_width=b-a,token_nucleotides=row.sequence[a:b]))
    return targets


def target_logits(bundle, ids, target, cache=False, replacements=None, residual=None):
    """Patch native pre-projection contexts, always cleaning hooks in finally."""
    modules=list(bundle.hf_model.bert.encoder.layer)
    captured, handles={},[]
    def hook(layer):
        def apply(module, inputs, output):
            if output.ndim!=2 or output.shape[0]!=len(ids):
                raise ValueError("Native context requires one unpadded aligned sequence")
            if cache:
                captured[layer]=output.detach().clone()
            if replacements is not None and layer in replacements:
                reference=replacements[layer]
                if reference.shape!=output.shape:
                    raise ValueError("Patch context shape mismatch")
                return reference.clone()
        return apply
    try:
        for layer, module in enumerate(modules):
            handles.append(module.attention.self.register_forward_hook(hook(layer)))
        if cache or residual is not None:
            def final_residual(module, inputs, output):
                if cache:
                    captured["final_residual"]=output.detach().clone()
                if residual is not None:
                    if residual.shape!=output.shape:
                        raise ValueError("Final residual shape mismatch")
                    return residual.clone()
            handles.append(modules[-1].register_forward_hook(final_residual))
        encoded=torch.tensor([ids],dtype=torch.long,device=bundle.device)
        with torch.no_grad():
            logits=bundle.hf_model(input_ids=encoded,attention_mask=torch.ones_like(encoded),return_dict=True).logits
        values=logits[0,target["index"]].detach()
        if not torch.isfinite(values).all():
            raise ValueError("Native patch produced nonfinite logits")
        return values,captured
    finally:
        for handle in handles:
            handle.remove()


def intervention_controls(bundle, pairs, tolerance=2e-4):
    rows=[]
    for entry in pairs:
        target=entry["target"]
        clean,clean_cache=target_logits(bundle,target["masked_ids"][0],target,cache=True)
        for role, index in (("motif",1),("sham",2)):
            edited,edited_cache=target_logits(bundle,target["masked_ids"][index],target,cache=True)
            identity,_=target_logits(bundle,target["masked_ids"][index],target,replacements=edited_cache)
            rescued,_=target_logits(bundle,target["masked_ids"][index],target,replacements=clean_cache)
            full_rescue,_=target_logits(bundle,target["masked_ids"][index],target,residual=clean_cache["final_residual"])
            reverse,_=target_logits(bundle,target["masked_ids"][0],target,residual=edited_cache["final_residual"])
            def difference(a,b):
                return float((a-b).abs().max().item())
            identity_error=difference(identity,edited)
            rescue_error=difference(full_rescue,clean)
            reverse_error=difference(reverse,edited)
            rows.append(dict(sequence_id=entry["pair"]["motif"]["sequence_id"],role=role,
                identity_max_logit_error=identity_error,rescue_max_logit_error=rescue_error,
                attention_only_rescue_max_logit_error=difference(rescued,clean),
                reverse_max_logit_error=reverse_error,
                passed=max(identity_error,rescue_error,reverse_error)<=tolerance,
                unpatched_target_effect=float((torch.log_softmax(clean,0)-torch.log_softmax(edited,0))[target["token_id"]]),
                scope="identity head contexts; final-residual rescue/reverse; wiring, not motif sensitivity"))
    if not all(r["passed"] for r in rows):
        raise ValueError("Native identity/whole-context rescue/reverse controls failed")
    return rows


def engineered_motif_fixture():
    """An explicit motif-dependent predictor uses the same context hook interface."""
    from types import SimpleNamespace
    class Context(torch.nn.Module):
        def forward(self, ids):
            output=torch.zeros(len(ids),2)
            output[:,0]=float(bool(torch.equal(ids[1:4],torch.tensor([1,2,3]))))
            return output
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            class Layer(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.context=Context()
                    self.attention=SimpleNamespace(self=self.context)
                def forward(self, ids):
                    return self.attention.self(ids)
            self.layer=Layer()
            self.bert=SimpleNamespace(encoder=SimpleNamespace(layer=[self.layer]))
        def forward(self,input_ids,**kwargs):
            contexts=self.bert.encoder.layer[0](input_ids[0])
            logits=torch.stack([contexts[:,1],contexts[:,0]*4],dim=-1)[None]
            return SimpleNamespace(logits=logits)
    bundle=SimpleNamespace(hf_model=Model(),device="cpu")
    target=dict(index=4,token_id=1)
    clean,cache=target_logits(bundle,[0,1,2,3,9,0],target,cache=True)
    edited,_=target_logits(bundle,[0,1,3,2,9,0],target)
    restored,_=target_logits(bundle,[0,1,3,2,9,0],target,replacements=cache)
    effect=float(torch.log_softmax(clean,0)[1]-torch.log_softmax(edited,0)[1])
    error=float((restored-clean).abs().max())
    if effect<=.5 or error>1e-6:
        raise ValueError("Engineered known-mechanism control failed")
    return dict(clean_minus_edited_log_probability=effect,rescue_max_logit_error=error,passed=True,
                scope="engineered computational sensitivity; not biological native-model sensitivity")


def influence_table(scores):
    clusters=genomic_clusters(scores.sequence_id.to_numpy(),sequences=scores.clean_sequence.to_numpy())
    result=scores.copy()
    result["cluster"]=clusters
    result["leave_cluster_out_mean"]=[float(scores.motif_minus_sham_loss[clusters!=c].mean()) for c in clusters]
    return result
