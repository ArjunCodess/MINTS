"""Run frozen natural-variant feasibility; failed gates prevent head search."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from dataclasses import replace
import json
import time

import numpy as np
import pandas as pd

from src.assay_stats import genomic_clusters
from src.config import DEFAULT_CONFIG
from src.controlled_edits import MotifDefinition
from src.motif_scoring import load_jaspar_ctcf_motif, motif_pssm
from src.native_endpoint import load_native_mlm
from src.native_followup import engineered_motif_fixture
from src.utils import sha256_file, write_json
from src.variant_assay import prepare_variant, score_variant
from src.variant_protocol import VariantProtocol, feasibility_gate, confirmation_readiness
from src.variant_statistics import cluster_summary, binding_agreement, select_head, simulate_cluster_power
from src.variant_quality import biological_qc

ROOT=Path(__file__).resolve().parents[1]


def run(output, input_path, device="auto"):
    output,input_path=Path(output).resolve(),Path(input_path).resolve()
    if (output/"protocol.json").exists():
        raise FileExistsError("A completed or attempted pilot requires a new output directory")
    output.mkdir(parents=True,exist_ok=True)
    protocol=VariantProtocol()
    names=["src/variant_assay.py","src/variant_protocol.py","src/variant_statistics.py","tools/run_variant_pilot.py",
           "src/native_endpoint.py","src/native_followup.py","src/assay_alignment.py","src/assay_stats.py",
           "src/modeling.py","src/config.py","src/motif_scoring.py","src/controlled_edits.py","src/utils.py","src/variant_quality.py"]
    sources={n:sha256_file(ROOT/n) for n in names}
    source_manifest=ROOT/"results/native_followup/allele_manifest.json"
    if input_path != ROOT/"results/native_followup/ctcf_allele_effects.csv":
        raise ValueError("This feasibility protocol accepts only the fixed published benchmark; new sources require a new protocol")
    prior=json.loads(source_manifest.read_text())
    if prior["table_sha256"] != sha256_file(input_path):
        raise ValueError("Biological source table does not match its extraction receipt")
    frozen=dict(**protocol.record(),model_revision=DEFAULT_CONFIG.model.revision,input_sha256=sha256_file(input_path),
        source_sha256=sources,biological_source_manifest_sha256=sha256_file(source_manifest),
        pilot_population="all 16 published rows; homozygous and unresolved adjacent variants excluded before model scoring",
        min_queries=1,coverage_scope="published pooled counts; phase, replicate and mapping-bias uncertainty unresolved",
        biological_power="not supplied by this small published benchmark")
    write_json(output/"protocol.json",frozen)
    started=time.time()
    receipt=dict(status="running",source_sha256=sources,protocol_sha256=sha256_file(output/"protocol.json"),command=sys.argv)
    write_json(output/"execution.json",receipt)
    try:
        table=pd.read_csv(input_path)
        group_sizes=table.groupby("locus_group").size().to_dict()
        write_json(output/"biological_qc.json",[dict(source_row=r["source_row"],**biological_qc(r)) for r in table.to_dict("records")])
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(DEFAULT_CONFIG.model.model_name,
            revision=DEFAULT_CONFIG.model.revision,trust_remote_code=True,local_files_only=DEFAULT_CONFIG.model.local_files_only)
        motif=MotifDefinition("CTCF",pssm=motif_pssm(load_jaspar_ctcf_motif()),fraction=.8)
        eligibility=[];cases=[];observed={};eligibility_diagnostics=[]
        for row in table.to_dict("records"):
            trace={}
            case,reason=prepare_variant(tokenizer,row,motif,protocol,group_sizes,diagnostics=trace)
            eligibility_diagnostics.append(dict(source_row=row["source_row"],reason=reason,**trace))
            eligibility.append(dict(source_row=row["source_row"],locus_group=row["locus_group"],reason=reason,retained=case is not None))
            if case is not None:
                cases.append(case)
                observed[case["variant_id"]]=float(row["log_odds_ratio"])
        pd.DataFrame(eligibility).to_csv(output/"eligibility.csv",index=False)
        write_json(output/"eligibility_diagnostics.json",eligibility_diagnostics)
        # Membership and controls are saved before any model inference.
        write_json(output/"membership.json",dict(cases=cases,inspection_role="exploratory published benchmark",
            biological_effects_used_for_selection=False))
        fixture=engineered_motif_fixture()
        write_json(output/"engineered_control.json",fixture)
        rows=[];diagnostics=[];bundle=None
        if cases:
            bundle,audit=load_native_mlm(replace(DEFAULT_CONFIG.model,device=device))
            write_json(output/"checkpoint.json",audit)
            for case in cases:
                result=score_variant(bundle,case,protocol)
                diagnostics.append(dict(variant_id=case["variant_id"],**result))
                rows.append(dict(variant_id=case["variant_id"],sequence_id=case["sequence_id"],locus_group=case["locus_group"],
                    variant_divergence=result["variant_divergence"],sham_divergence=result["sham_divergence"],
                    contrast=result["contrast"],binding_log_odds=observed[case["variant_id"]],pwm_delta=case["pwm_delta"],
                    controls_passed=result["controls_passed"],query_count=len(case["queries"])))
        else:
            write_json(output/"checkpoint.json",dict(status="not loaded; no sequence-eligible matched cases"))
        columns=["variant_id","sequence_id","locus_group","variant_divergence","sham_divergence","contrast",
                 "binding_log_odds","pwm_delta","controls_passed","query_count"]
        scores=pd.DataFrame(rows,columns=columns)
        scores.to_csv(output/"scores.csv",index=False)
        write_json(output/"query_diagnostics.json",diagnostics)
        preeligible=int(sum(r["zygosity"]=="heterozygous" and group_sizes[r["locus_group"]]==1 for r in table.to_dict("records")))
        summary=dict(scanned=len(table),singleton_heterozygous_loci=preeligible,retained=len(cases),
            retention=len(cases)/preeligible if preeligible else 0,controls_passed=all(r["controls_passed"] for r in rows) if rows else None,
            controls_status="passed" if rows else "not run; no eligible matched cases",
            engineered_control_passed=fixture["passed"],clusters=0,mean=None,ci_low=None,ci_high=None,
            binding_rho=None,binding_ci_low=None,binding_ci_high=None,selected_head=None,
            confirmation_status="not run; independent biological QC, membership and power required")
        if rows:
            clusters=genomic_clusters([r["sequence_id"] for r in rows],sequences=[c["reference_sequence"] for c in cases])
            scores["cluster"]=clusters;scores.to_csv(output/"scores.csv",index=False)
            summary.update(cluster_summary(scores.contrast,clusters,protocol.bootstrap_samples,protocol.seed))
            summary.update(binding_agreement(scores.variant_divergence,scores.binding_log_odds,clusters,protocol.bootstrap_samples,protocol.seed))
            pwm=binding_agreement(abs(scores.pwm_delta),scores.binding_log_odds,clusters,protocol.bootstrap_samples,protocol.seed)
            summary["pwm_magnitude_baseline"]=pwm
        decision=feasibility_gate(summary,protocol)
        write_json(output/"gate.json",decision)
        if decision["status"]=="eligible_for_discovery":
            head_rows=[]
            for layer,module in enumerate(bundle.hf_model.bert.encoder.layer):
                for head in range(module.attention.self.num_attention_heads):
                    effects=[score_variant(bundle,c,protocol,head=(layer,head))["head_contrast"] for c in cases]
                    estimate=cluster_summary(effects,clusters,protocol.bootstrap_samples,protocol.seed)
                    head_rows.append(dict(layer=layer,head=head,**estimate,effects=effects))
            write_json(output/"discovery_heads.json",head_rows)
            selected=select_head(head_rows)
            write_json(output/"selected_head.json",selected)
            summary["selected_head"]=[selected["layer"],selected["head"]]
            means=[np.mean(np.asarray(selected["effects"])[clusters==c]) for c in np.unique(clusters)]
            write_json(output/"power.json",simulate_cluster_power(means,protocol.meaningful_rescue_nats,
                retention=summary["retention"],seed=protocol.seed))
        else:
            write_json(output/"power.json",dict(status="not estimable",reason="pilot failed before head selection; discovery intervention variance unavailable",
                scenarios="normal approximation only; no sample-size authorization",
                illustrative_sd_to_clusters={str(sd):int(np.ceil(((1.96+.84)*sd/protocol.meaningful_rescue_nats)**2)) for sd in (.002,.005,.01)}))
        write_json(output/"summary.json",summary)
        readiness=confirmation_readiness(dict(pilot_passed=decision["status"]=="eligible_for_discovery"))
        write_json(output/"confirmation_readiness.json",readiness)
        receipt["status"]="completed"
        print(json.dumps(dict(**summary,decision=decision),indent=2))
        return summary
    except Exception as exc:
        receipt.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt.update(elapsed_seconds=time.time()-started,
            source_changed_during_run=any(sha256_file(ROOT/n)!=h for n,h in sources.items()),
            artifacts={p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name!="execution.json"})
        write_json(output/"execution.json",receipt)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/variant_pilot")
    parser.add_argument("--input",type=Path,default=ROOT/"results/native_followup/ctcf_allele_effects.csv")
    parser.add_argument("--device",default="auto")
    args=parser.parse_args();run(args.output,args.input,args.device)
