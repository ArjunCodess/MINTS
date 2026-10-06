"""Sequence-only natural substitution eligibility and native distribution assay."""
from collections import Counter
import math

import numpy as np
import torch

from .assay_alignment import exact_offsets, validate_patch_alignment
from .native_followup import target_geometry, target_logits


def validate_variant(row):
    clean, alt = row["reference_sequence"], row["alternate_sequence"]
    if row["genome_build"] not in ("hg19", "hg38"):
        raise ValueError("Explicit supported genome build required")
    if not clean or len(clean) != len(alt) or set(clean + alt) - set("ACGT"):
        raise ValueError("Canonical equal-length allele windows required")
    index = int(row["position_1based"]) - 1 - int(row["window_start_0based"])
    if not 0 <= index < len(clean) or int(row["window_end_0based"]) - int(row["window_start_0based"]) != len(clean):
        raise ValueError("Coordinate/window mismatch")
    differences = [i for i, (a, b) in enumerate(zip(clean, alt)) if a != b]
    if differences != [index] or clean[index] != row["reference"] or alt[index] != row["alternate"]:
        raise ValueError("Allele orientation or single-substitution mismatch")
    if str(row["reference_verified"]).lower() not in ("true", "1"):
        raise ValueError("Reference genome verification missing")
    for key in ("wgs_wt", "wgs_mt", "chip_wt", "chip_mt"):
        count = float(row[key])
        if not math.isfinite(count) or count < 0 or not count.is_integer():
            raise ValueError("Read counts must be finite nonnegative integers")
    return index


def prepare_variant(tokenizer, row, motif, protocol, group_sizes):
    """Choose controls and queries before inspecting any native model output."""
    index = validate_variant(row)
    if row["zygosity"] != "heterozygous" or min(row["wgs_wt"], row["wgs_mt"] ) <= 0:
        return None, "unidentifiable allele contrast"
    if group_sizes[row["locus_group"]] > 1:
        return None, "adjacent variants with unresolved phase"
    clean, alt = row["reference_sequence"], row["alternate_sequence"]
    try:
        alignment = validate_patch_alignment(tokenizer, clean, alt)
    except ValueError:
        return None, "reference/alternate BPE boundaries differ"
    offsets = alignment["clean_offsets"]
    encoded = [list(tokenizer(s, add_special_tokens=True, truncation=False)["input_ids"]) for s in (clean, alt)]
    hits = motif.hits(clean)
    covering = [h for h in hits if h[0] <= index < h[1]]
    if not covering:
        return None, "variant lacks threshold-passing reference CTCF motif"
    span = sorted(covering, key=lambda h: (-h[2], h[0]))[0]
    queries = []
    for i, (a, b) in enumerate(offsets):
        side, gap = target_geometry((index, index + 1), (a, b))
        if (b > a and side != "overlap" and gap <= protocol.query_radius_bp
                and encoded[0][i] == encoded[1][i]
                and not any(a < y and b > x for x,y,_ in hits)):
            queries.append(dict(index=i, span=[a, b], token_id=encoded[0][i], width=b-a))
    if not queries:
        return None, "no unchanged local query tokens"
    for query in queries:
        masked = [ids.copy() for ids in encoded]
        for ids in masked:
            ids[query["index"]] = tokenizer.mask_token_id
        if masked[0] == masked[1]:
            return None, "masking erases allele distinction"
    context = clean[max(0, index-1):index+2]
    edit_tokens = [b-a for a, b in offsets if a <= index < b]
    candidates = sorted(range(1, len(clean)-1), key=lambda j: (abs(j-index), j))
    for sham_index in candidates:
        if sham_index == index or abs(sham_index-index) > protocol.sham_radius_bp:
            continue
        if clean[sham_index-1:sham_index+2] != context:
            continue
        if any(a <= sham_index < b for a, b, _ in hits):
            continue
        if [b-a for a, b in offsets if a <= sham_index < b] != edit_tokens:
            continue
        selected_queries=[q for q in queries if q["span"][1] <= min(index,sham_index)
                          or q["span"][0] > max(index,sham_index)]
        if not selected_queries:
            continue
        if any(target_geometry((index,index+1), q["span"])[0] != target_geometry((sham_index,sham_index+1), q["span"])[0]
               or abs(target_geometry((index,index+1),q["span"])[1] - target_geometry((sham_index,sham_index+1),q["span"])[1]) > protocol.distance_tolerance_bp
               for q in selected_queries):
            continue
        def gc(j):
            window=clean[max(0,j-16):j+17]
            return (window.count("G")+window.count("C"))/len(window)
        if abs(gc(index)-gc(sham_index)) > protocol.gc_tolerance:
            continue
        sham = clean[:sham_index] + row["alternate"] + clean[sham_index+1:]
        if {(a,b) for a,b,_ in motif.hits(sham)} != {(a,b) for a,b,_ in hits}:
            continue
        if abs(motif.span_score(sham,span[0],span[1])-span[2]) > protocol.sham_pwm_tolerance_bits:
            continue
        try:
            validate_patch_alignment(tokenizer, clean, sham)
        except ValueError:
            continue
        sham_ids=list(tokenizer(sham,add_special_tokens=True,truncation=False)["input_ids"])
        if any(sham_ids[q["index"]] != q["token_id"] for q in selected_queries):
            continue
        return dict(variant_id=f"{row['chrom']}:{row['position_1based']}:{row['reference']}>{row['alternate']}",
            locus_group=row["locus_group"],sequence_id=f"{row['chrom']}:{row['window_start_0based']}-{row['window_end_0based']}",
            genome_build=row["genome_build"],reference_sequence=clean,alternate_sequence=alt,sham_sequence=sham,
            variant_index=index,sham_index=sham_index,queries=selected_queries,ids=[*encoded,sham_ids],
            substitution=row["reference"]+">"+row["alternate"],trinucleotide=context,
            pwm_delta=motif.span_score(alt,span[0],span[1])-span[2],motif_span=list(span[:2])), "retained"
    return None, "no sequence-only substitution/geometry matched sham"


def js_divergence(first, second):
    """Stable symmetric Jensen-Shannon divergence of native vocabulary logits."""
    if first.shape != second.shape or not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("Finite same-vocabulary logits required")
    a, b = torch.log_softmax(first.double(), -1), torch.log_softmax(second.double(), -1)
    mean = torch.logaddexp(a, b) - math.log(2)
    return float((.5*((a.exp()*(a-mean)).sum()+(b.exp()*(b-mean)).sum())).clamp(min=0))


def score_variant(bundle, case, protocol, head=None):
    """Fixed queries, identity/full-residual controls, optional one-head rescue."""
    rows=[]
    for query in case["queries"]:
        ids=[x.copy() for x in case["ids"]]
        for x in ids:
            x[query["index"]]=bundle.tokenizer.mask_token_id
        ref, cached=target_logits(bundle,ids[0],query,cache=True)
        identity, _=target_logits(bundle,ids[0],query,replacements=cached)
        identity_error=float((ref-identity).abs().max())
        values=[]
        for edited in ids[1:]:
            observed, _=target_logits(bundle,edited,query)
            restored, _=target_logits(bundle,edited,query,residual=cached["final_residual"])
            error=float((restored-ref).abs().max())
            if max(error,identity_error) > protocol.numerical_tolerance:
                raise ValueError("Native implementation control failed")
            effect=js_divergence(ref,observed)
            rescue=None
            if head is not None:
                layer, h=head
                module=bundle.hf_model.bert.encoder.layer[layer].attention.self
                width=module.attention_head_size
                def replace_head(module, inputs, output):
                    result=output.clone()
                    q=query["index"]
                    result[q,h*width:(h+1)*width]=cached[layer][q,h*width:(h+1)*width]
                    return result
                handle=module.register_forward_hook(replace_head)
                try:
                    patched,_=target_logits(bundle,edited,query)
                finally:
                    handle.remove()
                rescue=effect-js_divergence(ref,patched)
            values.append(dict(divergence=effect,full_rescue_error=error,head_rescue=rescue))
        rows.append(dict(query=query,identity_error=identity_error,variant=values[0],sham=values[1]))
    weights=np.array([r["query"]["width"] for r in rows],dtype=float)
    mean=lambda role,key:float(np.average([r[role][key] for r in rows],weights=weights))
    return dict(variant_divergence=mean("variant","divergence"),sham_divergence=mean("sham","divergence"),
        contrast=mean("variant","divergence")-mean("sham","divergence"),
        head_contrast=None if head is None else mean("variant","head_rescue")-mean("sham","head_rescue"),
        controls_passed=True,query_diagnostics=rows)
