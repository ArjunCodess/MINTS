"""Generate all numeric manuscript inserts from saved machine-readable sources."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from src.inference import bootstrap_mean_interval
from tools.review_tables import tex_table

root=Path(__file__).resolve().parents[1]
out=root/"results/review"
audit=json.loads((out/"correctness_audit.json").read_text())
counts=audit["motif_count_stages"]
metrics=pd.read_csv(out/"classification_metrics.csv")
(out/"table3.tex").write_text(tex_table(metrics,dict(gc="GC",kmer="$3$--$6$-mer",probe="Frozen readout")),encoding="utf-8")
(out/"table4.tex").write_text(tex_table(metrics,dict(probe="Full",historical_gc_probe="Historical matching",caliper_gc_probe="GC caliper")),encoding="utf-8")
names={"sequences":"CTCFSequences","unique_motif_support_tokens":"UniqueSupport","motif_hit_token_occurrences":"HitOccurrences",
       "finite_token_rows":"FiniteTokens","token_rows":"TokenRows","motif_absent_sequences":"AbsentSequences"}
macros=["\\newcommand{\\"+macro+"}{"+f"{counts[key]:,}"+"}" for key,macro in names.items()]
historical_pairs=audit["patching"]["promoter_tata"]
for macro,value in {"HistoricalTATAPairs":historical_pairs["pairs"],
                    "HistoricalTATATrain":historical_pairs["split_membership"]["train"],
                    "HistoricalTATATest":historical_pairs["split_membership"]["test"]}.items():
    macros.append("\\newcommand{\\"+macro+"}{"+str(value)+"}")
qk=pd.read_csv(root/"results/qk_alignment/ctcf_qk_alignment.csv")
enrich=pd.read_csv(root/"results/enrichment/ctcf_qk_alignment_matched_attention_enrichment.csv")
macros.extend([r"\newcommand{\BestQK}{"+f"{qk.pearson_r.max():.4f}"+"}",
               r"\newcommand{\BestEnrichment}{"+f"{enrich.rho.max():.4f}"+"}"])
(out/"numbers.tex").write_text("\n".join(macros)+"\n",encoding="utf-8")
lines=[r"\begin{tabular}{lrrrr}",r"\toprule",r"Task & Train positive & Train negative & Test positive & Test negative \\",r"\midrule"]
for task,report in audit["splits"].items():
    balance=report["class_counts"]
    label={"promoter_tata":"TATA promoter","promoter_no_tata":"Other promoter","splice_sites_donors":"Splice donor","splice_sites_acceptors":"Splice acceptor"}[task]
    lines.append(label+" & "+" & ".join(f"{balance[s][c]:,}" for s,c in [("train","1"),("train","0"),("test","1"),("test","0")])+r" \\")
(out/"dataset_table.tex").write_text("\n".join(lines+[r"\bottomrule",r"\end{tabular}"])+"\n",encoding="utf-8")
sweep=[]
for r in [.1,.2,.3,.4,.5]:
    for ratio in [1.1,1.25,1.5,2.]:
        joined=qk.merge(enrich,on=["layer","head"])
        sweep.append(dict(r_threshold=r,enrichment_threshold=ratio,heads=int(((joined.pearson_r>=r)&(joined.rho>=ratio)).sum())))
pd.DataFrame(sweep).to_csv(out/"ctcf_threshold_sweep.csv",index=False)
lines=[r"\begin{tabular}{lrrrr}",r"\toprule",r"QK $r$ cutoff & Ratio $1.1$ & Ratio $1.25$ & Ratio $1.5$ & Ratio $2.0$ \\",r"\midrule"]
for r in [.1,.2,.3,.4,.5]:
    lines.append(str(r)+" & "+" & ".join(str(row["heads"]) for row in sweep if row["r_threshold"]==r)+r" \\")
(out/"threshold_table.tex").write_text("\n".join(lines+[r"\bottomrule",r"\end{tabular}"])+"\n",encoding="utf-8")
patchroot=root/"results/cross_model/review_tata_heldout/patching"
table=pd.read_csv(patchroot/"promoter_tata_batch_dnabert_activation_patching.csv")
effects=np.load(patchroot/"promoter_tata_batch_dnabert_activation_patching_pair_effects.npz")
denominator=effects["denominator"]
best=table.sort_values("restoration",ascending=False).iloc[0]
summary=dict(pairs=len(denominator),best_layer=int(best.layer),best_head=int(best["head"]),best_mean=float(best.restoration),
             median=float(best.median_restoration),mean_ci=[float(best.mean_ci_low),float(best.mean_ci_high)],
             denominator_quantiles=np.quantile(denominator,[0,.25,.5,.75,1]).tolist(),
             negative_denominators=int((denominator<0).sum()),positive_denominators=int((denominator>0).sum()),
             denominators_near_zero=int((np.abs(denominator)<=1e-8).sum()),
             pm_greater_than_one=int((effects["restoration"]>1).sum()),
             min_absolute_denominator=float(np.abs(denominator).min()),
             selection_warning="largest observed mean selected over 144 heads; intervals are marginal and do not certify selected-head significance")
(out/"heldout_patching_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
lines=[r"\begin{tabular}{lrrrr}",r"\toprule",r"Layer/head & Mean PM & Median PM & Mean 95\% CI & Pairs \\",r"\midrule"]
for row in table.sort_values("restoration",ascending=False).head(3).itertuples():
    lines.append(f"{row.layer}/{row.head} & {row.restoration:.4f} & {row.median_restoration:.4f} & [{row.mean_ci_low:.4f}, {row.mean_ci_high:.4f}] & {row.pairs}"+r" \\")
(out/"patching_table.tex").write_text("\n".join(lines+[r"\bottomrule",r"\end{tabular}"])+"\n",encoding="utf-8")
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
                  denominator=denominator,selected_head_pm=values)).to_csv(out/"heldout_pair_plot_data.csv",index=False)
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
control=json.loads((out/"ctcf_native_controls_manifest.json").read_text())
positive=native[(native.mean_present_minus_absent>0)&(native.holm_p<.05)]
best_native=native.sort_values("mean_present_minus_absent",ascending=False).iloc[0]
native_macros={"NativePairs":control["valid_pairs"],"NativePositiveHeads":len(positive),
               "NativeBestLayer":int(best_native.layer),"NativeBestHead":int(best_native["head"])}
with (out/"numbers.tex").open("a",encoding="utf-8") as stream:
    for name,value in native_macros.items():
        stream.write("\\newcommand{\\"+name+"}{"+f"{value:,}"+"}\n")
    for name,value in [("NativeBestContrast",best_native.mean_present_minus_absent),
                       ("NativeBestLow",best_native.ci_low),("NativeBestHigh",best_native.ci_high),
                       ("NativeBestHolm",best_native.holm_p)]:
        stream.write("\\newcommand{\\"+name+"}{"+f"{value:.4f}"+"}\n")
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
manifest=json.loads((out/"tables_manifest.json").read_text())
manifest["artifacts"]={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in manifest["artifacts"]}
(out/"tables_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
