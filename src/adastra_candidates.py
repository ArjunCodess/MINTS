"""Structural import and sequence feasibility, without biological outcome selection."""
import hashlib

import pandas as pd

from .assay_alignment import validate_patch_alignment
from .native_followup import target_geometry


def import_candidates(table):
    if "#chr" in table and "chr" not in table:table=table.rename(columns={"#chr":"chr"})
    required={"chr","start","end","ID","ref","alt"}
    if not required<=set(table):raise ValueError("Missing ADASTRA genomic columns")
    records=[]
    for number,row in enumerate(table.to_dict("records"),1):
        start,end=str(row["start"]),str(row["end"])
        if not start.isdecimal() or not end.isdecimal() or int(end)!=int(start)+1:
            raise ValueError(f"Invalid BED-like SNV coordinates at row {number}")
        chrom=str(row["chr"])
        if not chrom.startswith("chr"):chrom="chr"+chrom
        if chrom not in {f"chr{i}" for i in range(1,23)}|{"chrX","chrY"}:
            raise ValueError(f"Unsupported hg38 chromosome at row {number}")
        ref,alt=str(row["ref"]),str(row["alt"])
        if ref not in "ACGT" or alt not in "ACGT" or len(ref)!=1 or len(alt)!=1 or ref==alt:
            raise ValueError(f"Invalid single substitution at row {number}")
        records.append(dict(source_row=number,variant_id=f"{chrom}:{end}:{ref}>{alt}",
            chrom=chrom,position_1based=int(end),reference=ref,alternate=alt,
            rs_id=str(row["ID"]),genome_build="hg38"))
    result=pd.DataFrame(records)
    if result.empty or result.variant_id.duplicated().any():
        raise ValueError("Candidate table must be nonempty with unique allele-specific coordinates")
    return result


def select_candidates(table,limit=4096,seed=1731):
    if type(limit) is not int or limit<1 or type(seed) is not int or seed<0:
        raise ValueError("Positive integer limit and nonnegative integer seed required")
    ranked=table.copy()
    ranked["selection_hash"]=[hashlib.sha256(f"{seed}:{v}".encode()).hexdigest() for v in ranked.variant_id]
    return ranked.sort_values(["selection_hash","variant_id"]).head(limit).reset_index(drop=True)


def allele_window(genome,row,width=204):
    if type(width) is not int or width<4 or width%2:raise ValueError("Even window width required")
    position=int(row["position_1based"])-1;start=position-width//2;end=start+width
    if row["chrom"] not in genome or start<0 or end>len(genome[row["chrom"]]):
        return None,"reference window outside chromosome"
    clean=str(genome[row["chrom"]][start:end]).upper()
    if len(clean)!=width or set(clean)-set("ACGT"):return None,"noncanonical reference window"
    index=position-start
    if clean[index]!=row["reference"]:return None,"hg38 reference allele mismatch"
    alt=clean[:index]+row["alternate"]+clean[index+1:]
    return dict(reference_sequence=clean,alternate_sequence=alt,variant_index=index,
                window_start_0based=start,window_end_0based=end),"reference verified"


def sequence_control(tokenizer,window,motif,protocol):
    """Apply the v2 sequence constraints without inventing genotypes or read counts."""
    clean,alt,index=window["reference_sequence"],window["alternate_sequence"],window["variant_index"]
    trace=dict(candidate_positions_checked=0,candidate_rejections={})
    def reject(reason):
        trace["candidate_rejections"][reason]=trace["candidate_rejections"].get(reason,0)+1
    try:offsets=validate_patch_alignment(tokenizer,clean,alt)["clean_offsets"]
    except ValueError:return None,"reference/alternate BPE boundaries differ",trace
    hits=motif.hits(clean);covering=[h for h in hits if h[0]<=index<h[1]]
    if not covering:return None,"variant lacks reference CTCF motif",trace
    span=sorted(covering,key=lambda h:(-h[2],h[0]))[0]
    ids=[list(tokenizer(s,add_special_tokens=True,truncation=False)["input_ids"]) for s in (clean,alt)]
    queries=[dict(index=i,span=[a,b],token_id=ids[0][i],width=b-a) for i,(a,b) in enumerate(offsets)
             if b>a and target_geometry((index,index+1),(a,b))[0]!="overlap"
             and target_geometry((index,index+1),(a,b))[1]<=protocol.query_radius_bp
             and ids[0][i]==ids[1][i] and not any(a<y and b>x for x,y,_ in hits)]
    if not queries:return None,"no unchanged local query tokens",trace
    context=clean[index-1:index+2];edit_width=[b-a for a,b in offsets if a<=index<b]
    for sham_index in sorted(range(1,len(clean)-1),key=lambda j:(abs(j-index),j)):
        if sham_index==index or abs(sham_index-index)>protocol.sham_radius_bp:continue
        trace["candidate_positions_checked"]+=1
        if clean[sham_index-1:sham_index+2]!=context:reject("trinucleotide mismatch");continue
        if any(a<=sham_index<b for a,b,_ in hits):reject("inside reference motif");continue
        if [b-a for a,b in offsets if a<=sham_index<b]!=edit_width:reject("edit-token width mismatch");continue
        chosen=[q for q in queries if q["span"][1]<=min(index,sham_index) or q["span"][0]>max(index,sham_index)]
        if not chosen:reject("no query outside both edits");continue
        if any(target_geometry((index,index+1),q["span"])[0]!=target_geometry((sham_index,sham_index+1),q["span"])[0]
               or abs(target_geometry((index,index+1),q["span"])[1]-target_geometry((sham_index,sham_index+1),q["span"])[1])>protocol.distance_tolerance_bp for q in chosen):
            reject("query distance or side mismatch");continue
        def gc(j):
            s=clean[max(0,j-16):j+17];return (s.count("G")+s.count("C"))/len(s)
        if abs(gc(index)-gc(sham_index))>protocol.gc_tolerance:reject("local GC mismatch");continue
        sham=clean[:sham_index]+alt[index]+clean[sham_index+1:]
        if {(a,b) for a,b,_ in motif.hits(sham)}!={(a,b) for a,b,_ in hits}:reject("motif-hit locations changed");continue
        if abs(motif.span_score(sham,span[0],span[1])-span[2])>protocol.sham_pwm_tolerance_bits:reject("motif score changed");continue
        try:validate_patch_alignment(tokenizer,clean,sham)
        except ValueError:reject("sham BPE boundaries differ");continue
        sham_ids=list(tokenizer(sham,add_special_tokens=True,truncation=False)["input_ids"])
        if any(sham_ids[q["index"]]!=q["token_id"] for q in chosen):reject("query token identity changed");continue
        triplet=[*ids,sham_ids]
        if any(len({tuple(x[:q["index"]]+[tokenizer.mask_token_id]+x[q["index"]+1:]) for x in triplet})<3 for q in chosen):
            reject("masking erases an edit");continue
        return dict(sham_index=sham_index,sham_sequence=sham,queries=chosen,ids=triplet),"sequence control feasible",trace
    return None,"no sequence-only substitution/geometry matched sham",trace
