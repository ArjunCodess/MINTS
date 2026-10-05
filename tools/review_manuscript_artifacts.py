"""Generate all numeric manuscript inserts from saved machine-readable sources."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import re
import argparse
import hashlib
import shutil
import numpy as np
import pandas as pd
from src.inference import bootstrap_mean_interval
from src.inference import validate_metric
from tools.review_tables import tex_table
from src.config import DEFAULT_CONFIG

root=Path(__file__).resolve().parents[1]
out=root/"results/review"

def import_completed_run(base):
    """Copy only completed scientific outputs, retaining their origin hashes."""
    base=Path(base).resolve()
    completed=set()
    for path in (base/"results").glob("pipeline_run*.json"):
        run=json.loads(path.read_text())
        completed.update(step["name"] for step in run.get("steps",[]) if step["status"]=="completed")
    required={"systematic_causal_intervention","sequence_genomic_controls","sequence_classifier_intervals"}
    if not required<=completed:
        raise ValueError(f"Cannot import incomplete scientific run; missing {sorted(required-completed)}")
    config=json.loads((base/"results/manifests/pipeline_config.json").read_text())["config"]
    if (config["model"]["model_name"]!=DEFAULT_CONFIG.model.model_name or
        config["model"]["revision"]!=DEFAULT_CONFIG.model.revision or
        config["data"]["hf_dataset_revision"]!=DEFAULT_CONFIG.data.hf_dataset_revision):
        raise ValueError("Imported run does not use the manuscript's pinned model and dataset")
    split_audit=json.loads((out/"correctness_audit.json").read_text())["splits"]
    imported=pd.read_csv(base/"results/review/classification_metrics.csv")
    for task,report in split_audit.items():
        full=imported[(imported.task==task)&(imported.method=="probe")]
        if len(full)!=1 or int(full.iloc[0].train_examples)!=sum(report["class_counts"]["train"].values()) or int(full.iloc[0].test_examples)!=sum(report["class_counts"]["test"].values()):
            raise ValueError(f"Imported {task} does not cover the audited full partitions")
    patchbase=root/"results/cross_model/review_tata_heldout"
    names=["classification_metrics.csv","gc_matching_diagnostics.json","tables_manifest.json",
           "ctcf_genomic_control_pairs.csv","ctcf_native_control_scores.npz",
           "ctcf_native_control_inference.csv","ctcf_sequence_qk_inference.csv","ctcf_native_controls_manifest.json"]
    names += [f"{task}_predictions.csv" for task in ["promoter_tata","promoter_no_tata","splice_sites_donors","splice_sites_acceptors"]]
    files=[(base/"results/review"/name,out/name) for name in names]
    stem="promoter_tata_batch_dnabert_activation_patching"
    files.append((base/"results/patching"/(stem+"_pair_effects.npz"),patchbase/"patching"/(stem+"_pair_effects.npz")))
    files.append((base/"results/figures"/(stem+"_heatmap.png"),patchbase/"figures"/(stem+"_heatmap.png")))
    files.append((base/"results/patching"/(stem+"_sequence_cluster_summary.csv"),patchbase/"patching"/(stem+".csv")))
    files.append((base/"results/patching"/(stem+"_sequence_cluster_summary.json"),patchbase/"patching"/(stem+"_sequence_cluster_summary.json")))
    for folder,name in [("counterfactuals","promoter_tata_batch_activation_patching_pairs.tsv"),
                        ("manifests",stem+"_patching_manifest.json"),
                        ("manifests","promoter_tata_batch_activation_patching_pairs_counterfactuals_manifest.json")]:
        files.append((base/"results"/folder/name,patchbase/folder/name))
    origin={destination.relative_to(root).as_posix():{"source":str(source),"sha256":hashlib.sha256(source.read_bytes()).hexdigest()}
            for source,destination in files}
    model_path=base/"results/manifests/model_hooked_encoder_manifest.json"
    model=json.loads(model_path.read_text())
    model.update(source_manifest=str(model_path),source_manifest_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                 revision=config["model"]["revision"],seed=config["data"]["seed"],
                 training_cache_sha256=hashlib.sha256((base/"results/activations/promoter_tata_train_residual_mean.npz").read_bytes()).hexdigest(),
                 scalar="trained standardized logistic probe decision function",pooling="attention-mask mean including special tokens")
    for source,destination in files:
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
    (patchbase/"review_model_manifest.json").write_text(json.dumps(model,indent=2))
    (out/"clean_result_import.json").write_text(json.dumps(dict(command=[sys.executable,*sys.argv],
        input_run=str(base),completed_required_steps=sorted(required),files=origin,
        scope="fresh classifier, held-out TATA patching and native controls; historical screen tables remain historical"),indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run-directory",help="Import saved completed-run results before generating the paper")
    args=parser.parse_args()
    if args.input_run_directory:
        import_completed_run(args.input_run_directory)

audit=json.loads((out/"correctness_audit.json").read_text())
references=json.loads((out/"reference_verification.json").read_text())
if hashlib.sha256((root/"paper/references.bib").read_bytes()).hexdigest()!=references["bibliography_sha256"]:
    raise ValueError("Bibliography changed after reference verification")
bib_keys=set(re.findall(r"@\w+\{([^,]+),",(root/"paper/references.bib").read_text()))
if bib_keys!={reference["key"] for reference in references["references"]}:
    raise ValueError("Bibliography differs from the verified reference list")
counts=audit["motif_count_stages"]
metrics=pd.read_csv(out/"classification_metrics.csv")
for column in ["auroc","auroc_ci_low","auroc_ci_high"]:
    validate_metric("auroc",metrics[column])
tables={"PerformanceTable":tex_table(metrics,dict(gc="GC",kmer="$3$--$6$-mer",probe="Frozen readout")),
        "ControlTable":tex_table(metrics,dict(probe="Full",historical_gc_probe="Historical matching",caliper_gc_probe="GC caliper"))}
names={"sequences":"CTCFSequences","unique_motif_support_tokens":"UniqueSupport","motif_hit_token_occurrences":"HitOccurrences",
       "finite_token_rows":"FiniteTokens","token_rows":"TokenRows","motif_absent_sequences":"AbsentSequences"}
macros=["\\newcommand{\\"+macro+"}{"+f"{counts[key]:,}"+"}" for key,macro in names.items()]
for task,macro in {"promoter_tata":"CaliperTATACount", "promoter_no_tata":"CaliperOtherCount",
                   "splice_sites_donors":"CaliperDonorCount", "splice_sites_acceptors":"CaliperAcceptorCount"}.items():
    retained=metrics[(metrics.task==task)&(metrics.method=="caliper_gc_probe")].iloc[0]
    macros.append("\\newcommand{\\"+macro+"}{"+f"{int(retained.test_examples):,}"+"}")
historical_pairs=audit["patching"]["promoter_tata"]
for macro,value in {"HistoricalTATAPairs":historical_pairs["pairs"],
                    "HistoricalTATATrain":historical_pairs["split_membership"]["train"],
                    "HistoricalTATATest":historical_pairs["split_membership"]["test"]}.items():
    macros.append("\\newcommand{\\"+macro+"}{"+str(value)+"}")
qk=pd.read_csv(root/"results/qk_alignment/ctcf_qk_alignment.csv")
enrich=pd.read_csv(root/"results/enrichment/ctcf_qk_alignment_matched_attention_enrichment.csv")
macros.extend([r"\newcommand{\BestQK}{"+f"{qk.pearson_r.max():.4f}"+"}",
               r"\newcommand{\BestEnrichment}{"+f"{enrich.rho.max():.4f}"+"}"])
lines=[r"\begin{tabular}{lrrrr}",r"\toprule",r"Task & Train positive & Train negative & Test positive & Test negative \\",r"\midrule"]
for task,report in audit["splits"].items():
    balance=report["class_counts"]
    label={"promoter_tata":"TATA promoter","promoter_no_tata":"Other promoter","splice_sites_donors":"Splice donor","splice_sites_acceptors":"Splice acceptor"}[task]
    lines.append(label+" & "+" & ".join(f"{balance[s][c]:,}" for s,c in [("train","1"),("train","0"),("test","1"),("test","0")])+r" \\")
tables["DatasetTable"]="\n".join(lines+[r"\bottomrule",r"\end{tabular}"])
sweep=[]
for r in [.1,.2,.3,.4,.5]:
    for ratio in [1.1,1.25,1.5,2.]:
        joined=qk.merge(enrich,on=["layer","head"])
        sweep.append(dict(r_threshold=r,enrichment_threshold=ratio,heads=int(((joined.pearson_r>=r)&(joined.rho>=ratio)).sum())))
pd.DataFrame(sweep).to_csv(out/"ctcf_threshold_sweep.csv",index=False)
lines=[r"\begin{tabular}{lrrrr}",r"\toprule",r"QK $r$ cutoff & Ratio $1.1$ & Ratio $1.25$ & Ratio $1.5$ & Ratio $2.0$ \\",r"\midrule"]
for r in [.1,.2,.3,.4,.5]:
    lines.append(str(r)+" & "+" & ".join(str(row["heads"]) for row in sweep if row["r_threshold"]==r)+r" \\")
tables["ThresholdTable"]="\n".join(lines+[r"\bottomrule",r"\end{tabular}"])
patchroot=root/"results/cross_model/review_tata_heldout/patching"
table=pd.read_csv(patchroot/"promoter_tata_batch_dnabert_activation_patching.csv")
effects=np.load(patchroot/"promoter_tata_batch_dnabert_activation_patching_pair_effects.npz")
denominator=effects["denominator"]
macros.append("\\newcommand{\\HeldoutTATAPairs}{"+str(len(denominator))+"}")
best=table.sort_values("restoration",ascending=False).iloc[0]
summary=dict(pairs=len(denominator),best_layer=int(best.layer),best_head=int(best["head"]),best_mean=float(best.restoration),
             median=float(best.median_restoration),mean_ci=[float(best.mean_ci_low),float(best.mean_ci_high)],
             denominator_quantiles=np.quantile(denominator,[0,.25,.5,.75,1]).tolist(),
             negative_denominators=int((denominator<0).sum()),positive_denominators=int((denominator>0).sum()),
             denominators_near_zero=int((np.abs(denominator)<=1e-8).sum()),
             pm_greater_than_one=int((effects["restoration"]>1).sum()),
             min_absolute_denominator=float(np.abs(denominator).min()),
             selection_warning="largest observed mean selected over 144 heads; intervals are marginal and do not certify selected-head significance")
overshoot=effects["restoration"]>1
summary["overshoots_with_positive_denominator"]=int(overshoot[denominator>0].sum())
summary["overshoots_with_negative_denominator"]=int(overshoot[denominator<0].sum())
summary["overshoot_investigation"]="No denominator is within 1e-8 of zero; sign-changing reference differences and heterogeneous absolute effects make mean PM unstable. Off-manifold geometry was not measured, so overshoot is not evidence of motif-specific causality."
(out/"heldout_patching_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
lines=[r"\begin{tabular}{lrrrr}",r"\toprule",r"Layer/head & Mean PM & Median PM & Mean 95\% CI & Pairs \\",r"\midrule"]
for row in table.sort_values("restoration",ascending=False).head(3).itertuples():
    lines.append(f"{row.layer}/{row.head} & {row.restoration:.4f} & {row.median_restoration:.4f} & [{row.mean_ci_low:.4f}, {row.mean_ci_high:.4f}] & {row.pairs}"+r" \\")
tables["PatchingTable"]="\n".join(lines+[r"\bottomrule",r"\end{tabular}"])
# Figure source retains every pair, including overshoots and sign-sensitive effects.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
values=effects["restoration"][:,int(best.layer),int(best["head"])]
absolute_effect=values*denominator
summary["selected_head_absolute_effect_quantiles"]=np.quantile(absolute_effect,[0,.25,.5,.75,1]).tolist()
summary["selected_head_absolute_effect_mean_ci"]=bootstrap_mean_interval(absolute_effect)
(out/"heldout_patching_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
pd.DataFrame(dict(sequence_id=effects["sequence_ids"],clean_score=effects["clean_scores"],corrupted_score=effects["corrupted_scores"],
                  denominator=denominator,selected_head_pm=values,overshooting_heads=overshoot.sum(axis=(1,2)),
                  minimum_pm=np.nanmin(effects["restoration"],axis=(1,2)),maximum_pm=np.nanmax(effects["restoration"],axis=(1,2)))).to_csv(out/"heldout_pair_plot_data.csv",index=False)
fig,axes=plt.subplots(1,2,figsize=(8,3))
axes[0].scatter(np.arange(len(values)),values,color="#246179")
axes[0].axhline(1,color="grey",linestyle="--")
axes[0].axhline(0,color="grey",linewidth=.5)
axes[0].set(xlabel="Held-out pair index",ylabel="Selected-head PM",title="Exploratory maximum over heads")
axes[1].scatter(denominator,values,color="#246179")
axes[1].axvline(0,color="grey",linewidth=.5)
axes[1].set(xlabel="Clean minus corrupted probe score",ylabel="Selected-head PM",title="Denominator sensitivity")
fig.tight_layout()
fig.savefig(out/"heldout_patching.png",dpi=180)
plt.close(fig)

# Native attention associations are a separate, observational assay.
native=pd.read_csv(out/"ctcf_native_control_inference.csv")
qk_sequence=pd.read_csv(out/"ctcf_sequence_qk_inference.csv")
validate_metric("pearson_r",qk_sequence.mean_present_r)
validate_metric("p_value",qk_sequence.holm_p)
control=json.loads((out/"ctcf_native_controls_manifest.json").read_text())
positive=native[(native.mean_present_minus_absent>0)&(native.holm_p<.05)]
best_native=native.sort_values("mean_present_minus_absent",ascending=False).iloc[0]
native_macros={"NativePairs":control["valid_pairs"],"NativePositiveHeads":len(positive),
               "NativeBestLayer":int(best_native.layer),"NativeBestHead":int(best_native["head"])}
for name,value in native_macros.items():
    macros.append("\\newcommand{\\"+name+"}{"+f"{value:,}"+"}")
for name,value in [("NativeBestContrast",best_native.mean_present_minus_absent),
                   ("NativeBestLow",best_native.ci_low),("NativeBestHigh",best_native.ci_high),
                   ("NativeBestHolm",best_native.holm_p)]:
    macros.append("\\newcommand{\\"+name+"}{"+f"{value:.4f}"+"}")
best_qk=qk_sequence.sort_values("mean_present_r",ascending=False).iloc[0]
for name,value in {"SequenceQKPairs":control["qk_valid_pairs"],"SequenceQKUndefined":control["qk_undefined_pairs"],
                   "SequenceQKHighHeads":int((qk_sequence.mean_present_r>=.5).sum()),
                   "SequenceQKBestLayer":int(best_qk.layer),"SequenceQKBestHead":int(best_qk["head"])}.items():
    macros.append("\\newcommand{\\"+name+"}{"+f"{value:,}"+"}")
for name,value in [("SequenceQKBestMean",best_qk.mean_present_r),("SequenceQKBestLow",best_qk.present_ci_low),
                   ("SequenceQKBestHigh",best_qk.present_ci_high)]:
    macros.append("\\newcommand{\\"+name+"}{"+f"{value:.4f}"+"}")
lines=[r"\begin{tabular}{lrrr}",r"\toprule",
       r"Layer/head & Mean present $r$ [95\% CI] & Present--absent [95\% CI] & Holm $p$ \\",r"\midrule"]
for row in qk_sequence.sort_values("mean_present_r",ascending=False).head(3).itertuples():
    lines.append(f"{row.layer}/{row.head} & {row.mean_present_r:.4f} [{row.present_ci_low:.4f}, {row.present_ci_high:.4f}] & "
                 f"{row.mean_present_minus_absent:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}] & {row.holm_p:.4f}"+r" \\")
tables["SequenceQKTable"]="\n".join(lines+[r"\bottomrule",r"\end{tabular}"])
display=native.sort_values("mean_present_minus_absent",ascending=False).head(8).iloc[::-1]
fig,axis=plt.subplots(figsize=(7,3))
y=np.arange(len(display))
axis.errorbar(display.mean_present_minus_absent,y,
              xerr=np.array([display.mean_present_minus_absent-display.ci_low,
                             display.ci_high-display.mean_present_minus_absent]),fmt="o",color="#246179")
axis.set_yticks(y,[f"{r.layer}/{r.head}" for r in display.itertuples()])
axis.axvline(0,color="grey",linewidth=.5)
axis.set(xlabel="Present minus absent native-attention density",ylabel="Layer/head")
fig.tight_layout()
fig.savefig(out/"ctcf_native_controls.png",dpi=180)
plt.close(fig)
import hashlib
tex_path=root/"paper/results.tex"
header="% Generated from saved results by tools/review_manuscript_artifacts.py.\n"
definitions=["\\newcommand{\\"+name+"}{%\n"+body.rstrip()+"\n}" for name,body in tables.items()]
tex_path.write_text(header+"\n".join(macros)+"\n\n"+"\n\n".join(definitions)+"\n",encoding="utf-8",newline="\n")
manifest=json.loads((out/"tables_manifest.json").read_text())
manifest["artifacts"]={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in manifest["artifacts"] if not name.endswith(".tex")}
manifest["paper_results"]={"path":"paper/results.tex","sha256":hashlib.sha256(tex_path.read_bytes()).hexdigest(),
                         "command":"python tools/review_manuscript_artifacts.py"}
sources=[out/"correctness_audit.json",out/"classification_metrics.csv",
         root/"results/qk_alignment/ctcf_qk_alignment.csv",
         root/"results/enrichment/ctcf_qk_alignment_matched_attention_enrichment.csv",
         patchroot/"promoter_tata_batch_dnabert_activation_patching.csv",
         patchroot/"promoter_tata_batch_dnabert_activation_patching_pair_effects.npz",
         out/"ctcf_native_control_inference.csv",out/"ctcf_native_controls_manifest.json",
         out/"ctcf_sequence_qk_inference.csv",out/"pinned_dataset_comparison.json",out/"upstream_dataset_snapshot.json",
         out/"reference_verification.json",root/"paper/references.bib"]
if (out/"clean_result_import.json").exists():
    sources.append(out/"clean_result_import.json")
for name in ("clean_reproduction.json", "clean_input_comparison.json", "clean_feature_comparison.json", "clean_screen_comparison.json", "donor_sequence_cluster_summary.json"):
    if (out/name).exists():
        sources.append(out/name)
if (patchroot/"promoter_tata_batch_dnabert_activation_patching_sequence_cluster_summary.json").exists():
    sources.append(patchroot/"promoter_tata_batch_dnabert_activation_patching_sequence_cluster_summary.json")
if (root/"paper/hardened_results.tex").exists():
    sources.extend([root/"paper/hardened_results.tex",root/"results/hardened/paper_manifest.json",root/"paper/main.tex"])
manifest["paper_sources"]={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
manifest["figures"]={
    "results/review/ctcf_native_controls.png":{
        "sha256":hashlib.sha256((out/"ctcf_native_controls.png").read_bytes()).hexdigest(),
        "sources":["results/review/ctcf_native_control_inference.csv"]},
    "results/review/heldout_patching.png":{
        "sha256":hashlib.sha256((out/"heldout_patching.png").read_bytes()).hexdigest(),
        "sources":[(patchroot/"promoter_tata_batch_dnabert_activation_patching_pair_effects.npz").relative_to(root).as_posix(),
                   (patchroot/"promoter_tata_batch_dnabert_activation_patching.csv").relative_to(root).as_posix()]},
}
(out/"tables_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
claim_audit={
    "generated_by":"python tools/review_manuscript_artifacts.py",
    "claims":[
        dict(claim="Fixed-classifier held-out prediction",status="supported with sequence-bootstrap intervals",
             evidence=["classification_metrics.csv","*_predictions.csv"],boundary="no training/checkpoint variability or causal mechanism"),
        dict(claim="Historical GC equality and motif-count reconciliation",status="reproduced",
             evidence=["correctness_audit.json","gc_matching_diagnostics.json"],boundary="historical promoter matching is the full membership reordered"),
        dict(claim="Disjoint stored partitions",status="supported under specified checks",
             evidence=["correctness_audit.json","*_membership.jsonl"],boundary="exact/RC, coordinate overlap, chromosome and equal-length <=2 substitutions; no indel/homology/pretraining guarantee"),
        dict(claim="No head passed historical motif-local screens",status="descriptive only",
             evidence=["ctcf_threshold_sweep.csv"],boundary="weight-only reconstruction; invalid token-level significance withdrawn"),
        dict(claim="Native attention and sequence-QK conditional associations",status="sequence-pair inference completed",
             evidence=["ctcf_native_control_inference.csv","ctcf_sequence_qk_inference.csv","ctcf_native_control_scores.npz","ctcf_native_controls_manifest.json"],
             boundary="separate Holm families; observational exchangeability; marginal displayed-head intervals"),
        dict(claim="Held-out TATA probe restoration",status="historical diagnostic; withdrawn as aligned intervention evidence",
             evidence=["heldout_patching_summary.json","heldout_pair_plot_data.csv"],boundary="ten selected test pairs; trained probe; composition edits; six of ten pairs have verified offset mismatch"),
        dict(claim="No biological motif detector exists",status="removed",evidence=[],boundary="screen failure cannot establish absence"),
        dict(claim="Native CTCF causal effect",status="not claimed",evidence=[],boundary="no validated native CTCF output"),
        dict(claim="Cross-model ranking or Spearman rho above one",status="removed",evidence=["correctness_audit.json"],boundary="stored rho is an enrichment ratio; NT comparison not validated"),
        dict(claim="Preregistered thresholds",status="removed",evidence=[],boundary="heuristic thresholds; no timestamped preregistration evidence"),
    ],
    "source_hashes":manifest["paper_sources"],
    "remaining_limitations":["Independent upstream label reconstruction is incomplete","Native CTCF causality is not established",
                             "Clean reproduction completed through recorded resumptions; earlier failures and source changes are preserved"]}
claim_audit["review_concerns"]=[
    dict(concern=concern,evidence=evidence,code=code,manuscript=paragraph,status=status)
    for concern,evidence,code,paragraph,status in [
        ("Duplicate Seq-head/Readout and Table 3/4 provenance",["correctness_audit.json","tables_manifest.json"],
         ["src/task_performance.py","tools/review_tables.py","tools/review_manuscript_artifacts.py"],"Predictive comparisons and result tables","corrected; single frozen readout"),
        ("Full versus GC-matched AUROC equality",["gc_matching_diagnostics.json","*_predictions.csv"],
         ["src/probing.py","tools/review_tables.py"],"Composition matching changes the promoter result","membership equality explained; caliper population regenerated"),
        ("281915 versus 256918 motif counts",["correctness_audit.json","clean_input_comparison.json"],
         ["src/motif_scoring.py","tools/review_audit.py"],"Nucleotide-to-token alignment and CTCF count stages","unique support versus hit occurrences reconciled"),
        ("Invalid Nucleotide Transformer comparison",["correctness_audit.json"],
         ["src/reproduce.py","src/cli.py"],"Interpretation and limitations","comparison removed; no cross-model claim"),
        ("327 patching pairs versus 212 test examples",["correctness_audit.json","heldout_patching_summary.json"],
         ["src/patching.py","tools/review_patching.py"],"Patching is a held-out probe experiment","historical pooling explained; test-only pairs enforced"),
        ("Old tables and submission remnants",["command_final63_tests.log"],
         ["tests/test_review_integrity.py"],"Rewritten manuscript; old appendices removed","source scan enforced; no PDF inspection"),
        ("Dataset sources, partitions, duplication and leakage",["*_membership.jsonl","pinned_dataset_comparison.json","upstream_dataset_snapshot.json"],
         ["src/data_ingestion.py","src/integrity.py"],"Datasets and audit scope","analyzed membership documented and checked; independent upstream label reconstruction remains limited"),
        ("CTCF assay definitions and motif-absent controls",["ctcf_native_controls_manifest.json","ctcf_genomic_control_pairs.csv"],
         ["tools/review_ctcf_controls.py"],"Historical assay, native attention and sequence-level content QK","implemented; observational assumptions stated"),
        ("Sequence-level uncertainty and 144-head selection",["ctcf_native_control_inference.csv","ctcf_sequence_qk_inference.csv","classification_metrics.csv"],
         ["src/inference.py","tools/review_ctcf_controls.py"],"Uncertainty and selection and main tables","pair/sequence resampling and separate Holm families implemented"),
        ("Registration claims and threshold robustness",["ctcf_threshold_sweep.csv"],
         ["src/threshold_sensitivity.py","tools/review_manuscript_artifacts.py"],"Evidence and output targets and threshold sweep","heuristic status stated; invalid null interpretations withdrawn"),
        ("Incomplete CTCF causal evidence",[],["src/reproduce.py"],"Conclusion and limitations",
         "conclusion narrowed to specified screen failure; no validated native CTCF causal target"),
        ("Patching scalar, denominator distributions and PM overshoot",["heldout_patching_summary.json","heldout_pair_plot_data.csv"],
         ["src/patching.py","tools/review_manuscript_artifacts.py"],"Patching procedure and results","probe target and exploratory uncertainty explicit; off-manifold geometry unmeasured"),
        ("K-mer superiority and choice of 3-to-6-mers",["classification_metrics.csv"],
         ["tools/review_tables.py"],"Predictive comparisons and Prediction is context","point advantage stated; heuristic range and fixed-classifier uncertainty stated"),
        ("Motivation, biology, theory and claim boundaries",["claim_evidence_audit.json"],
         ["paper/main.tex"],"Abstract, introduction, methods and limitations","rewritten; conditional theory and cross-model claims moderated"),
        ("Licenses and research declarations",["LICENSE","upstream_dataset_snapshot.json"],
         ["paper/main.tex"],"Availability, licenses, and declarations","author-reported funding/conflicts included; unverified terms and AI history disclosed"),
        ("Clean reproducibility, commands and failures",["commands.jsonl","clean_reproduction.json","clean_input_comparison.json","clean_cuda_retry_exit.json"],
         ["tools/review_full_pipeline.py","src/reproduce.py"],"Availability and limitations","all 13 required stages completed after recorded repairs and resumptions; failures retained"),
    ]
]
claim_audit["anonymization_scope"]="Manuscript source is anonymous. The repository MIT copyright names the author and Git preserves provenance; no anonymous source archive is claimed."
(out/"claim_evidence_audit.json").write_text(json.dumps(claim_audit,indent=2),encoding="utf-8")
