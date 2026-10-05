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
    for key in ("position_1based","window_start_0based","window_end_0based"):
        number=float(row[key])
        if isinstance(row[key],bool) or not math.isfinite(number) or not number.is_integer() or number<0:
            raise ValueError("Genomic coordinates must be finite nonnegative integers")
    if int(row["position_1based"])<1:
        raise ValueError("Source position must be one-based")
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


def prepare_variant(tokenizer, row, motif, protocol, group_sizes, diagnostics=None):
    """Choose controls and queries before inspecting any native model output."""
    index = validate_variant(row)
    trace=diagnostics if diagnostics is not None else {}
    trace.update(variant_index=index,candidate_rejections={},candidate_positions_checked=0)
    def reject(reason):
        trace["candidate_rejections"][reason]=trace["candidate_rejections"].get(reason,0)+1
    if row["zygosity"] != "heterozygous" or min(row["wgs_wt"], row["wgs_mt"] ) <= 0:
        return None, "unidentifiable allele contrast"
    if group_sizes[row["locus_group"]] > 1:
        return None, "adjacent variants with unresolved phase"
    clean, alt = row["reference_sequence"], row["alternate_sequence"]
    try:
        alignment = validate_patch_alignment(tokenizer, clean, alt)
    except ValueError:
        a,b=exact_offsets(tokenizer,clean),exact_offsets(tokenizer,alt)
        trace["reference_token_count"]=len(a)
        trace["alternate_token_count"]=len(b)
        trace["reference_variant_tokens"]=[list(span) for span in a if span[0]<=index<span[1]]
        trace["alternate_variant_tokens"]=[list(span) for span in b if span[0]<=index<span[1]]
        trace["first_boundary_difference"]=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
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
    trace.update(initial_query_count=len(queries),trinucleotide=context,motif_span=list(span[:2]))
    edit_tokens = [b-a for a, b in offsets if a <= index < b]
    candidates = sorted(range(1, len(clean)-1), key=lambda j: (abs(j-index), j))
    for sham_index in candidates:
        if sham_index == index or abs(sham_index-index) > protocol.sham_radius_bp:
            continue
        trace["candidate_positions_checked"]+=1
        if clean[sham_index-1:sham_index+2] != context:
            reject("trinucleotide mismatch")
            continue
        if any(a <= sham_index < b for a, b, _ in hits):
            reject("inside reference motif")
            continue
        if [b-a for a, b in offsets if a <= sham_index < b] != edit_tokens:
            reject("edit-token width mismatch")
            continue
        selected_queries=[q for q in queries if q["span"][1] <= min(index,sham_index)
                          or q["span"][0] > max(index,sham_index)]
        if not selected_queries:
            reject("no query outside both edits")
            continue
        if any(target_geometry((index,index+1), q["span"])[0] != target_geometry((sham_index,sham_index+1), q["span"])[0]
               or abs(target_geometry((index,index+1),q["span"])[1] - target_geometry((sham_index,sham_index+1),q["span"])[1]) > protocol.distance_tolerance_bp
               for q in selected_queries):
            reject("query distance or side mismatch")
            continue
        def gc(j):
            window=clean[max(0,j-16):j+17]
            return (window.count("G")+window.count("C"))/len(window)
        if abs(gc(index)-gc(sham_index)) > protocol.gc_tolerance:
            reject("local GC mismatch")
            continue
        sham = clean[:sham_index] + row["alternate"] + clean[sham_index+1:]
        if {(a,b) for a,b,_ in motif.hits(sham)} != {(a,b) for a,b,_ in hits}:
            reject("motif-hit locations changed")
            continue
        if abs(motif.span_score(sham,span[0],span[1])-span[2]) > protocol.sham_pwm_tolerance_bits:
            reject("motif score changed")
            continue
        try:
            validate_patch_alignment(tokenizer, clean, sham)
        except ValueError:
            reject("sham BPE boundaries differ")
            continue
        sham_ids=list(tokenizer(sham,add_special_tokens=True,truncation=False)["input_ids"])
        if any(sham_ids[q["index"]] != q["token_id"] for q in selected_queries):
            reject("query token identity changed")
            continue
        trace.update(selected_sham=sham_index,selected_query_count=len(selected_queries))
        return dict(variant_id=f"{row['chrom']}:{row['position_1based']}:{row['reference']}>{row['alternate']}",
            locus_group=row["locus_group"],sequence_id=f"{row['chrom']}:{row['window_start_0based']}-{row['window_end_0based']}",
            genome_build=row["genome_build"],reference_sequence=clean,alternate_sequence=alt,sham_sequence=sham,
            variant_index=index,sham_index=sham_index,queries=selected_queries,ids=[*encoded,sham_ids],
            substitution=row["reference"]+">"+row["alternate"],trinucleotide=context,
            pwm_delta=motif.span_score(alt,span[0],span[1])-span[2],motif_span=list(span[:2])), "retained"
    return None, "no sequence-only substitution/geometry matched sham"


def js_divergence(first, second):
    """Stable symmetric Jensen-Shannon divergence of native vocabulary logits."""
    if (first.ndim != 1 or first.numel() < 2 or first.shape != second.shape
            or not torch.isfinite(first).all() or not torch.isfinite(second).all()):
        raise ValueError("Finite same-vocabulary logits required")
    a, b = torch.log_softmax(first.double(), -1), torch.log_softmax(second.double(), -1)
    mean = torch.logaddexp(a, b) - math.log(2)
    return float((.5*((a.exp()*(a-mean)).sum()+(b.exp()*(b-mean)).sum())).clamp(min=0))


def score_variant(bundle, case, protocol, head=None):
    """Fixed queries, identity/full-residual controls, optional one-head rescue."""
    if len(case.get("ids",[]))!=3 or not case.get("queries"):
        raise ValueError("One aligned reference/alternate/sham triplet and nonempty queries required")
    lengths={len(ids) for ids in case["ids"]}
    if len(lengths)!=1:
        raise ValueError("Intervention inputs must have equal token lengths")
    for q in case["queries"]:
        if type(q["index"]) is not int or not 0 <= q["index"] < len(case["ids"][0]) or type(q["width"]) is not int or q["width"]<=0:
            raise ValueError("Invalid query index or nucleotide weight")
        if any(ids[q["index"]]!=q["token_id"] for ids in case["ids"]):
            raise ValueError("Query token identity differs across intervention inputs")
        masked=[ids[:q["index"]]+[bundle.tokenizer.mask_token_id]+ids[q["index"]+1:] for ids in case["ids"]]
        if masked[0]==masked[1] or masked[0]==masked[2]:
            raise ValueError("Masking erases variant or sham distinction")
    if head is not None:
        if len(head)!=2 or any(type(index) is not int for index in head):
            raise ValueError("Layer/head selection must contain two integers")
        layer,h=head;layers=bundle.hf_model.bert.encoder.layer
        if not 0 <= layer < len(layers) or not 0 <= h < layers[layer].attention.self.num_attention_heads:
            raise ValueError("Selected head is outside the model")
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
