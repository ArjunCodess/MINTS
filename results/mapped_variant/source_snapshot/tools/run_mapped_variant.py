"""Execute the separately frozen mapped-query exploratory sensitivity study."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter
from dataclasses import replace
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
from pyfaidx import Fasta
import torch
from transformers import AutoTokenizer
from scipy.stats import spearmanr, ttest_1samp

from src.adastra_candidates import allele_window
from src.assay_stats import genomic_clusters
from src.config import DEFAULT_CONFIG
from src.control_diagnostics import token_map, common_queries, same_side_queries, diagnose_window
from src.controlled_edits import MotifDefinition
from src.mapped_variant import score_mapped_case
from src.motif_scoring import load_jaspar_ctcf_motif,motif_pssm,find_jaspar_matrix_path
from src.native_endpoint import load_native_mlm
from src.utils import write_json,sha256_file,set_reproducibility_seed
from src.variant_protocol import VariantProtocol
from src.variant_statistics import cluster_summary

ROOT=Path(__file__).resolve().parents[1]


def run(output,device):
    if output.exists():raise FileExistsError("Use a fresh study directory")
    if shutil.disk_usage(ROOT).free<2*1024**3:raise OSError("Need 2 GiB spare disk")
    output.mkdir(parents=True)
    diagnostics=ROOT/"results/control_diagnostics"
    sources=[p for p in (ROOT/"src").glob("*.py")]+[Path(__file__).resolve()]
    hashes={p.relative_to(ROOT).as_posix():sha256_file(p) for p in sources}
    membership=pd.read_csv(diagnostics/"membership.csv")
    membership.to_csv(output/"membership.csv",index=False,lineterminator="\n")
    protocol=dict(version="mapped-query-js-v1",role="exploratory single-SNV reference scenarios",
        design_document_sha256=sha256_file(ROOT/"docs/mapped_variant_protocol.md"),
        membership_sha256=sha256_file(output/"membership.csv"),diagnostic_execution_sha256=sha256_file(diagnostics/"execution.json"),
        selection="all diagnostic members, same hash order; no outcome selection",source_records=512556,
        model_revision=DEFAULT_CONFIG.model.revision,primary="equal-cluster mean variant-minus-sham width-weighted JS in nats",
        seed=1731,bootstrap_samples=10000,block_bp=1000000,min_clusters=8,numerical_tolerance=0.0002,
        query="identical nucleotide span and token identity across triplet; separate masks; mapped indices",
        changes=["complete BPE alignment removed","edit-token width matching removed"],
        causal_gate="native lower CI > 0; all implementation controls pass; >=8 full-boundary-stable clusters; biological QC",
        stopping="stop before head search on any causal gate failure; no confirmation without independent frozen population",
        source_sha256=hashes,device=device,precision="float32; float64 JS; TF32 off",disk_free_bytes=shutil.disk_usage(ROOT).free)
    write_json(output/"protocol.json",protocol)
    shutil.copyfile(ROOT/"docs/mapped_variant_protocol.md",output/"design.md")
    snapshot=output/"source_snapshot"
    for name in hashes:
        destination=snapshot/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    start=time.time()
    receipt=dict(status="running",started_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        source_sha256=hashes,protocol_sha256=sha256_file(output/"protocol.json"),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in
            ("torch","transformers","tokenizers","numpy","pandas","scipy","pyfaidx")},command=sys.argv)
    write_json(output/"execution.json",receipt)
    try:
        token=AutoTokenizer.from_pretrained(DEFAULT_CONFIG.model.model_name,revision=DEFAULT_CONFIG.model.revision,
            trust_remote_code=True,local_files_only=True)
        motif=MotifDefinition("CTCF",pssm=motif_pssm(load_jaspar_ctcf_motif()),fraction=.8)
        candidates=pd.read_csv(diagnostics/"candidate_predicates.csv.gz")
        selected=candidates[candidates.local_no_width].copy()
        controls={v:g.sort_values(["distance_bp","sham_index"]).iloc[0].to_dict() for v,g in selected.groupby("variant_id")}
        rows=[];cases=[]
        with Fasta(str(ROOT/"data/genomes/hg38.fa"),as_raw=True,sequence_always_upper=True,rebuild=False) as genome:
            for r in membership.to_dict("records"):
                window,reason=allele_window(genome,r)
                motif_ok=False;query_ok=False;case=None
                if window:
                    clean,alt,index=(window[k] for k in ("reference_sequence","alternate_sequence","variant_index"))
                    maps=[token_map(token,s) for s in (clean,alt)]
                    hits=motif.hits(clean);covering=sorted([h for h in hits if h[0]<=index<h[1]],key=lambda h:(-h[2],h[0]))
                    motif_ok=bool(covering)
                    query_ok=bool(common_queries(maps,index,hits))
                    reason="absent reference motif" if not motif_ok else "absent common allele query" if not query_ok else "no matched sham"
                    if r["variant_id"] in controls:
                        summary,checks=diagnose_window(token,window,motif,VariantProtocol())
                        check=next(c for c in checks if c["sham_index"]==int(controls[r["variant_id"]]["sham_index"]))
                        if not check["local_no_width"]:raise ValueError("Frozen diagnostic control could not be reproduced")
                        j=check["sham_index"];sham=clean[:j]+alt[index]+clean[j+1:]
                        maps.append(token_map(token,sham))
                        queries=same_side_queries(common_queries(maps,index,hits),index,j)
                        case=dict(**r,**window,sham_index=j,sham_sequence=sham,queries=queries,
                            ids=[m[1] for m in maps],motif_span=list(covering[0][:2]),
                            pwm_delta=motif.span_score(alt,*covering[0][:2])-covering[0][2],
                            edit_width_matched=check["edit_width"],full_boundary_stable=maps[0][0]==maps[1][0]==maps[2][0],
                            token_counts=[len(m[1]) for m in maps],query_index_shift=any(len(set(q["indices"]))>1 for q in queries),
                            control_diagnostics=check,biological_qc_eligible=None)
                        cases.append(case);reason="matched sequence control"
                rows.append(dict(variant_id=r["variant_id"],reference_verified=window is not None,
                    motif_eligible=motif_ok,correspondence_eligible=motif_ok and query_ok,matched_control=case is not None,reason=reason))
        pd.DataFrame(rows).to_csv(output/"eligibility.csv",index=False,lineterminator="\n")
        names=np.array([[f"{r['chrom']}:{r['window_start_0based']}-{r['window_end_0based']}"]*3 for r in cases])
        sequences=np.array([[r[k] for k in ("reference_sequence","alternate_sequence","sham_sequence")] for r in cases])
        clusters=genomic_clusters(names,block_bp=1000000,sequences=sequences)
        for r,c in zip(cases,clusters,strict=True):r["cluster"]=int(c)
        write_json(output/"cases.json",cases)
        write_json(output/"inspection_ledger.json",dict(role="exploratory only",native_outcomes_opened=False,
            membership_sha256=sha256_file(output/"membership.csv"),case_membership_sha256=sha256_file(output/"cases.json"),
            biological_outcomes_used=False,confirmation_assigned=False))
        if len(set(clusters))<8:raise ValueError("Sequence feasibility gate failed: fewer than eight clusters")
        set_reproducibility_seed(1731)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        bundle,loading=load_native_mlm(replace(DEFAULT_CONFIG.model,device=device,local_files_only=True))
        from huggingface_hub import snapshot_download
        cache=Path(snapshot_download(DEFAULT_CONFIG.model.model_name,revision=DEFAULT_CONFIG.model.revision,local_files_only=True))
        model_files={p.name:sha256_file(p) for p in cache.iterdir() if p.is_file()}
        write_json(output/"model.json",dict(**loading,files_sha256=model_files,
            tokenizer_sha256=sha256_file(cache/"tokenizer.json"),precision=str(next(bundle.hf_model.parameters()).dtype),
            device_name=torch.cuda.get_device_name(0) if device.startswith("cuda") else platform.processor(),
            genome_sha256=sha256_file(ROOT/"data/genomes/hg38.fa"),motif_sha256=sha256_file(find_jaspar_matrix_path())))
        raw_dir=ROOT/"data/adastra/mapped_native_v1"
        raw_dir.mkdir(parents=True,exist_ok=False)
        scores=[]
        for number,case in enumerate(cases,1):
            path=raw_dir/f"case_{number:04d}.npz"
            value=score_mapped_case(bundle,case,save_logits=path)
            scores.append(dict(variant_id=case["variant_id"],cluster=case["cluster"],**value,
                raw_logits_path=path.relative_to(ROOT).as_posix(),raw_logits_sha256=sha256_file(path)))
            write_json(output/"scores.json",scores)
            print(f"scored {number}/{len(cases)}: contrast={value['contrast']:.6g}",flush=True)
        frame=pd.DataFrame([{**{k:c[k] for k in ("variant_id","chrom","cluster","pwm_delta","edit_width_matched","full_boundary_stable","query_index_shift")},
            **{k:s[k] for k in ("contrast","variant_divergence","sham_divergence","implementation_valid")},
            "reference_tokens":c["token_counts"][0],"alternate_token_change":c["token_counts"][1]-c["token_counts"][0],
            "sham_token_change":c["token_counts"][2]-c["token_counts"][0]} for c,s in zip(cases,scores,strict=True)])
        frame.to_csv(output/"case_scores.csv",index=False,lineterminator="\n")
        grouped=frame.groupby("cluster").agg(contrast=("contrast","mean"),variant=("variant_divergence","mean"),
            sham=("sham_divergence","mean"),absolute_pwm=("pwm_delta",lambda x:abs(x).mean()),cases=("variant_id","size"))
        grouped.to_csv(output/"cluster_scores.csv",lineterminator="\n")
        primary=cluster_summary(frame.contrast,frame.cluster,repetitions=10000)
        width=frame[frame.edit_width_matched]
        width_summary=cluster_summary(width.contrast,width.cluster,repetitions=10000)
        rho,p=spearmanr(grouped.variant,grouped.absolute_pwm)
        width_means=width.groupby("cluster").contrast.mean()
        width_p=float(ttest_1samp(width_means,0).pvalue) if len(width_means)>1 else None
        family=[dict(test="absolute PWM magnitude vs variant JS",effect=float(rho),p_raw=float(p)),
                dict(test="width-matched contrast",effect=width_summary["mean"],p_raw=width_p)]
        finite=[i for i,r in enumerate(family) if r["p_raw"] is not None and np.isfinite(r["p_raw"])]
        ordered=sorted(finite,key=lambda i:family[i]["p_raw"]);previous=0.
        for rank,i in enumerate(ordered):
            previous=max(previous,min(1.,family[i]["p_raw"]*(len(family)-rank)))
            family[i]["p_holm"]=previous
        stable_clusters=int(frame[frame.full_boundary_stable].cluster.nunique())
        primary_pass=primary["ci_low"] is not None and primary["ci_low"]>0
        gates=dict(sequence_feasibility=True,implementation_valid=True,native_sensitivity=primary_pass,
                   tokenization_isolation=stable_clusters>=8,biological_qc=False,head_search=False,confirmation=False)
        summary=dict(source_records=512556,selected=len(rows),reference_verified=sum(r["reference_verified"] for r in rows),
            motif_eligible=sum(r["motif_eligible"] for r in rows),correspondence_eligible=sum(r["correspondence_eligible"] for r in rows),
            matched_controls=len(cases),implementation_valid=len(scores),biological_qc_cases=None,independent_donors=None,
            genomic_proxy_clusters=len(grouped),full_boundary_stable_cases=int(frame.full_boundary_stable.sum()),
            full_boundary_stable_clusters=stable_clusters,width_matched_cases=len(width),primary=primary,
            width_matched=width_summary,secondary_tests=family,gates=gates,
            leave_cluster_out=[dict(omitted_cluster=int(c),mean=float(grouped.drop(c).contrast.mean())) for c in grouped.index],
            chromosome_sensitivity=cluster_summary(frame.contrast,frame.chrom,repetitions=10000),
            query_index_shift_cases=int(frame.query_index_shift.sum()),
            stop_reasons=[k for k in ("native_sensitivity","tokenization_isolation","biological_qc") if not gates[k]],
            head_search="not run; causal gates failed",confirmation="not run; no selected head or independent frozen cohort")
        write_json(output/"summary.json",summary)
        ledger=json.loads((output/"inspection_ledger.json").read_text());ledger["native_outcomes_opened"]=True
        write_json(output/"inspection_ledger.json",ledger)
        receipt["status"]="completed"
        print(json.dumps(summary,indent=2))
    except Exception as exc:
        receipt.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        receipt.update(elapsed_seconds=time.time()-start,source_changed_during_run=any(sha256_file(ROOT/n)!=h for n,h in hashes.items()),
            artifacts={p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name!="execution.json"})
        write_json(output/"execution.json",receipt)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"results/mapped_variant")
    p.add_argument("--device",default="cuda")
    a=p.parse_args();run(a.output,a.device)
