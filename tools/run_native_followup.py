"""Execute recovery calibration, geometry feasibility, and intervention controls."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from collections import Counter
from dataclasses import replace
import argparse
import json
import subprocess
import time

import numpy as np
import pandas as pd

from src.assay_alignment import exact_offsets
from src.config import DEFAULT_CONFIG
from src.controlled_edits import MotifDefinition
from src.motif_scoring import load_jaspar_ctcf_motif,motif_pssm
from src.native_endpoint import NativeProtocol,load_native_mlm,native_score
from src.native_followup import (rematch_sham,recovery_targets,intervention_controls,
    engineered_motif_fixture,influence_table,target_geometry)
from src.utils import write_json,sha256_file

ROOT=Path(__file__).resolve().parents[1]


def run(output,device="auto"):
    output=Path(output).resolve()
    if (output/"protocol.json").exists():
        raise FileExistsError("Use fresh outputs for revised exploratory protocols")
    output.mkdir(parents=True,exist_ok=True)
    sources=["src/native_followup.py","tools/run_native_followup.py","src/native_endpoint.py",
             "src/assay_alignment.py","src/assay_stats.py","src/controlled_edits.py",
             "src/modeling.py","src/config.py","src/motif_scoring.py","src/utils.py"]
    digests={name:sha256_file(ROOT/name) for name in sources}
    input_path=DEFAULT_CONFIG.paths.ctcf_dir/"ctcf_gm12878_sequences.tsv"
    old=ROOT/"results/native_endpoint"
    protocol=dict(analysis_status="exploratory follow-up, not fresh confirmation",seed=1730,
        geometry=dict(fixed_original_target=True,fixed_original_motif_edit=True,same_side=True,
                      distance_tolerance_bp=2,window_width_and_token_budget="exact",
                      infeasible_controls="exclude; no tolerance relaxation or target search"),
        calibration=dict(chromosomes=["chr16","chr17"],sequence_cap=128,
                         token_selection="one uniformly random real token per seeded fixed window",
                         selection_uses_model_output=False,eligibility="canonical DNA only",
                         frequency_chromosomes=["chr14","chr15"],frequency_window_cap=512,
                         frequency_estimator="empirical token counts plus-one over full vocabulary",
                         interpretation="native recovery competence, not biological confirmation"),
        intervention=dict(population="all 12 saved eligible pilot pairs",identity_tolerance=2e-4,
            positive_control="replace entire final residual to include embedding/residual bypass",
            attention_only="descriptive; need not recover clean due to residual bypass",
            head_search=False),
        model_revision=DEFAULT_CONFIG.model.revision,input_sha256=sha256_file(input_path),
        prior_pairs_sha256=sha256_file(old/"pairs.jsonl"),source_sha256=digests)
    write_json(output/"protocol.json",protocol)
    receipt=dict(status="running",protocol_sha256=sha256_file(output/"protocol.json"),source_sha256=digests,
                 started_utc=pd.Timestamp.now(tz="UTC").isoformat(),command=sys.argv)
    write_json(output/"execution.json",receipt)
    started=time.time()
    try:
        bundle,audit=load_native_mlm(replace(DEFAULT_CONFIG.model,device=device,local_files_only=True))
        write_json(output/"checkpoint.json",audit)
        pairs=[json.loads(line) for line in (old/"pairs.jsonl").read_text().splitlines()]
        motif=MotifDefinition("CTCF",pssm=motif_pssm(load_jaspar_ctcf_motif()),fraction=.8)
        geometry=[]
        retained=[]
        for entry in pairs:
            matched,decision=rematch_sham(bundle.tokenizer,entry["pair"],motif,NativeProtocol(),2)
            record=entry["pair"]["motif"]
            sham=entry["pair"]["sham"]
            side,gap=target_geometry((record["start"],record["end"]),entry["target"]["span"])
            sham_side,sham_gap=target_geometry((sham["start"],sham["end"]),entry["target"]["span"])
            geometry.append(dict(sequence_id=record["sequence_id"],motif_side=side,motif_gap_bp=gap,
                                 sham_side=sham_side,sham_gap_bp=sham_gap,**decision))
            if matched is not None:
                retained.append(matched)
        pd.DataFrame(geometry).to_csv(output/"geometry.csv",index=False)
        if retained:
            # Outcomes stay explicit exploratory evidence, even if geometry is feasible.
            effect_rows=[]
            for entry in retained:
                scores=[native_score(bundle,ids,entry["target"])["log_probability"] for ids in entry["target"]["masked_ids"]]
                effect_rows.append(dict(sequence_id=entry["pair"]["motif"]["sequence_id"],
                                       clean=scores[0],motif=scores[1],sham=scores[2],contrast=scores[2]-scores[1]))
            pd.DataFrame(effect_rows).to_csv(output/"geometry_scores.csv",index=False)
        table=pd.read_csv(input_path,sep="\t")
        reference=table[table.chrom.isin(["chr14","chr15"])].sort_values(["chrom","start","end"])
        reference=reference.iloc[np.random.default_rng(1730).permutation(len(reference))[:512]]
        frequencies=Counter()
        frequency_members=[]
        for row in reference.itertuples():
            frequencies.update(i for i in bundle.tokenizer(row.sequence,add_special_tokens=False)["input_ids"]
                               if i not in bundle.tokenizer.all_special_ids)
            frequency_members.append(f"{row.chrom}:{row.start}-{row.end}")
        vocab=bundle.hf_model.config.vocab_size
        total=sum(frequencies.values())+vocab
        targets=recovery_targets(bundle.tokenizer,table,["chr16","chr17"],128,1730)
        rows=[]
        for index,entry in enumerate(targets):
            score=native_score(bundle,entry["masked_ids"],entry["target"])
            count=frequencies[entry["target"]["token_id"]]
            rows.append(dict(sequence_id=entry["sequence_id"],sequence=entry["sequence"],
                target_index=entry["target"]["index"],target_token_id=entry["target"]["token_id"],
                target_start=entry["target"]["span"][0],target_end=entry["target"]["span"][1],
                token_width=entry["token_width"],token_nucleotides=entry["token_nucleotides"],
                frequency_count=count,frequency_bin="0" if count==0 else "1-9" if count<10 else "10+",
                frequency_baseline_log_probability=float(np.log((count+1)/total)),**score))
            if (index+1)%32==0:
                print(f"recovery calibration: {index+1}/{len(targets)}",flush=True)
        calibration=pd.DataFrame(rows)
        calibration["reciprocal_rank"]=1/calibration["rank"]
        calibration["top1"]=calibration["rank"]==1
        calibration["top5"]=calibration["rank"]<=5
        calibration["width_bin"]=pd.cut(calibration.token_width,[0,3,6,np.inf],labels=["1-3","4-6","7+"]).astype(str)
        calibration.to_csv(output/"recovery_scores.csv",index=False)
        by_group=calibration.groupby(["width_bin","frequency_bin"],observed=True).agg(
            sequences=("sequence_id","size"),mean_log_probability=("log_probability","mean"),
            baseline_log_probability=("frequency_baseline_log_probability","mean"),
            median_rank=("rank","median"),mean_reciprocal_rank=("reciprocal_rank","mean"),
            top1_rate=("top1","mean"),top5_rate=("top5","mean"))
        by_group.to_csv(output/"recovery_strata.csv")
        write_json(output/"recovery_membership.json",dict(frequency=frequency_members,calibration=[r["sequence_id"] for r in rows],
                   membership_scope="disjoint chromosomes from original native pilot, not fresh binding confirmation"))
        print("validating native intervention controls",flush=True)
        controls=intervention_controls(bundle,pairs)
        pd.DataFrame(controls).to_csv(output/"intervention_controls.csv",index=False)
        fixture=engineered_motif_fixture()
        write_json(output/"engineered_control.json",fixture)
        influence=influence_table(pd.read_csv(old/"scores.csv"))
        influence.to_csv(output/"pilot_influence.csv",index=False)
        # Compare every scanned sequence with its retained/excluded status.
        lookup={f"{r.chrom}:{r.start}-{r.end}":r for r in table.itertuples()}
        selected=set(influence.sequence_id)
        eligibility=[]
        prior_audit=json.loads((old/"eligibility.json").read_text())
        for entry in prior_audit["entries"]:
            row=lookup[entry["sequence_id"]]
            offsets=exact_offsets(bundle.tokenizer,row.sequence)
            hits=motif.hits(row.sequence)
            eligibility.append(dict(sequence_id=entry["sequence_id"],retained=entry["sequence_id"] in selected,
                nucleotide_length=len(row.sequence),gc_fraction=(row.sequence.count("G")+row.sequence.count("C"))/len(row.sequence),
                motif_strength=max((h[2] for h in hits),default=np.nan),real_tokens=sum(b>a for a,b in offsets),
                mean_token_width=float(np.mean([b-a for a,b in offsets if b>a])),
                exclusion=entry.get("target_exclusion",entry.get("exclusion",""))))
        pd.DataFrame(eligibility).to_csv(output/"pilot_eligibility.csv",index=False)
        summary=dict(geometry_retained=len(retained),geometry_evaluated=len(geometry),geometry_status="infeasible" if not retained else "exploratory",
            calibration_sequences=len(calibration),median_rank=float(calibration["rank"].median()),
            mean_reciprocal_rank=float(calibration.reciprocal_rank.mean()),top1_rate=float(calibration.top1.mean()),
            mean_log_probability=float(calibration.log_probability.mean()),
            frequency_baseline_log_probability=float(calibration.frequency_baseline_log_probability.mean()),
            native_controls_passed=all(r["passed"] for r in controls),native_control_rows=len(controls),
            attention_only_rescue_median_error=float(np.median([r["attention_only_rescue_max_logit_error"] for r in controls])),
            engineered_control_passed=fixture["passed"],pilot_scanned=len(eligibility),pilot_retained=len(influence),
            leave_cluster_out_min=float(influence.leave_cluster_out_mean.min()),
            leave_cluster_out_max=float(influence.leave_cluster_out_mean.max()),
            claim_boundary="no head search; no fresh confirmation; descriptive/exploratory calibration")
        write_json(output/"summary.json",summary)
        receipt["status"]="completed"
        return summary
    except Exception as exc:
        receipt.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["elapsed_seconds"]=time.time()-started
        receipt["source_changed_during_run"]=any(sha256_file(ROOT/n)!=h for n,h in digests.items())
        receipt["artifacts"]={p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name!="execution.json"}
        write_json(output/"execution.json",receipt)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/native_followup")
    parser.add_argument("--device",default="auto")
    args=parser.parse_args()
    print(json.dumps(run(args.output,args.device),indent=2))
