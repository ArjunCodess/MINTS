"""Native MLM assay with explicit nucleotide-span query correspondence."""
import numpy as np
import torch

from .variant_assay import js_divergence


def mapped_logits(bundle, ids, query_index, cache=False, donor=None, head=None, final=False):
    """Transfer one mapped query row, never an equal-index entire tensor."""
    if not 0 <= query_index < len(ids) or ids[query_index] != bundle.tokenizer.mask_token_id:
        raise ValueError("Recipient must mask the declared query")
    layers = bundle.hf_model.bert.encoder.layer
    if donor is not None and not final and head is None:
        raise ValueError("A context transfer requires an explicit layer/head")
    if head is not None:
        layer,h=head
        if not 0 <= layer < len(layers) or not 0 <= h < layers[layer].attention.self.num_attention_heads:
            raise ValueError("Layer/head out of range")
    captured,handles={},[]
    def row(output):
        if output.ndim != 2 or output.shape[0] != len(ids):
            raise ValueError("Expected pinned-model single unpadded flattened context")
        return output[query_index]
    def context_hook(layer):
        def apply(module, inputs, output):
            value=row(output)
            if cache:
                captured[layer]=value.detach().clone()
            if donor is not None and not final and layer==head[0]:
                width=module.attention_head_size
                a,b=head[1]*width,(head[1]+1)*width
                if donor[layer].shape!=value.shape:
                    raise ValueError("Donor context channel dimension mismatch")
                result=output.clone()
                result[query_index,a:b]=donor[layer][a:b]
                return result
        return apply
    def final_hook(module,inputs,output):
        value=row(output)
        if cache:
            captured["final_query"]=value.detach().clone()
        if donor is not None and final:
            if donor["final_query"].shape!=value.shape:
                raise ValueError("Donor final representation channel mismatch")
            result=output.clone()
            result[query_index]=donor["final_query"]
            return result
    try:
        for layer,module in enumerate(layers):
            handles.append(module.attention.self.register_forward_hook(context_hook(layer)))
        handles.append(layers[-1].register_forward_hook(final_hook))
        encoded=torch.tensor([ids],device=bundle.device,dtype=torch.long)
        with torch.no_grad():
            logits=bundle.hf_model(input_ids=encoded,attention_mask=torch.ones_like(encoded),return_dict=True).logits
        if logits.ndim!=3 or logits.shape[:2]!=(1,len(ids)):
            raise ValueError("Native vocabulary extraction shape mismatch")
        values=logits[0,query_index].detach()
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite native distribution")
        normal=torch.softmax(values.double(),-1)
        if abs(float(normal.sum())-1)>1e-12:
            raise ValueError("Native distribution normalization failed")
        return values,captured
    finally:
        for handle in handles:
            handle.remove()


def validate_case(bundle,case):
    sequences=[case[k] for k in ("reference_sequence","alternate_sequence","sham_sequence")]
    index,j=case["variant_index"],case["sham_index"]
    if len({len(s) for s in sequences})!=1 or any(set(s)-set("ACGT") for s in sequences):
        raise ValueError("Canonical equal nucleotide length required")
    if [i for i,(a,b) in enumerate(zip(sequences[0],sequences[1])) if a!=b]!=[index]:
        raise ValueError("Variant difference must be exactly the declared SNV")
    if [i for i,(a,b) in enumerate(zip(sequences[0],sequences[2])) if a!=b]!=[j]:
        raise ValueError("Sham difference must be exactly its declared SNV")
    if sequences[0][j]!=sequences[0][index] or sequences[2][j]!=sequences[1][index]:
        raise ValueError("Sham substitution differs from variant")
    from .control_diagnostics import token_map
    maps=[token_map(bundle.tokenizer,s) for s in sequences]
    if case["ids"]!=[m[1] for m in maps]:
        raise ValueError("Saved token IDs differ from pinned tokenizer")
    if not case["queries"]:
        raise ValueError("Nonempty query set required")
    for q in case["queries"]:
        a,b=q["span"]
        if q["width"]!=b-a or b<=a or any(a<=k<b for k in (index,j)):
            raise ValueError("Invalid query span or masked edit")
        for m,i in zip(maps,q["indices"],strict=True):
            if m[0][i]!=(a,b) or m[1][i]!=q["token_id"]:
                raise ValueError("Query spans or identities disagree")
        if len(set(sequences[r][a:b] for r in range(3)))!=1:
            raise ValueError("Compared prediction tasks differ in nucleotide target")


def score_mapped_case(bundle,case,tolerance=2e-4,save_logits=None):
    validate_case(bundle,case)
    rows=[];native=[]
    for q in case["queries"]:
        ids=[x.copy() for x in case["ids"]]
        for x,i in zip(ids,q["indices"],strict=True):
            x[i]=bundle.tokenizer.mask_token_id
        if len({tuple(x) for x in ids})!=3:
            raise ValueError("Masking erased an allele or sham difference")
        values,caches=[],[]
        for x,i in zip(ids,q["indices"],strict=True):
            val,cache=mapped_logits(bundle,x,i,cache=True)
            values.append(val);caches.append(cache)
        native.append(np.stack([v.cpu().numpy() for v in values]))
        identity=[];restoration=[];repeat=[]
        for r,(x,i) in enumerate(zip(ids,q["indices"],strict=True)):
            ident,_=mapped_logits(bundle,x,i,donor=caches[r],head=(0,0))
            again,_=mapped_logits(bundle,x,i)
            identity.append(float((ident-values[r]).abs().max()))
            repeat.append(float((again-values[r]).abs().max()))
            if r:
                recovered,_=mapped_logits(bundle,x,i,donor=caches[0],final=True)
                reverse,_=mapped_logits(bundle,ids[0],q["indices"][0],donor=caches[r],final=True)
                restoration.extend([float((recovered-values[0]).abs().max()),float((reverse-values[r]).abs().max())])
        error=max(identity+restoration+repeat)
        if error>tolerance:
            raise ValueError(f"Mapped implementation controls failed: {error}")
        rows.append(dict(query=q,variant_js=js_divergence(values[0],values[1]),
            sham_js=js_divergence(values[0],values[2]),identity_errors=identity,
            final_restoration_errors=restoration,repeat_errors=repeat,
            normalization_error=max(abs(float(torch.softmax(v.double(),-1).sum())-1) for v in values)))
    weights=np.array([r["query"]["width"] for r in rows],dtype=float)
    variant=float(np.average([r["variant_js"] for r in rows],weights=weights))
    sham=float(np.average([r["sham_js"] for r in rows],weights=weights))
    if save_logits is not None:
        np.savez_compressed(save_logits,logits=np.stack(native),
            query_indices=np.array([q["indices"] for q in case["queries"]]),
            query_spans=np.array([q["span"] for q in case["queries"]]),weights=weights)
    return dict(variant_divergence=variant,sham_divergence=sham,contrast=variant-sham,
                implementation_valid=True,query_diagnostics=rows)
