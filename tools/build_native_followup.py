"""Verify follow-up evidence and generate influence/eligibility figures and text."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG
from src.motif_scoring import load_jaspar_ctcf_motif,motif_pssm,scan_sequence_with_pssm
from src.utils import write_json,sha256_file

ROOT=Path(__file__).resolve().parents[1]


def build(output=ROOT/"results/native_followup"):
    output=Path(output)
    receipt=json.loads((output/"execution.json").read_text())
    if receipt["status"]!="completed" or receipt["source_changed_during_run"]:
        raise ValueError("Follow-up execution incomplete or changed during run")
    for name,digest in receipt["source_sha256"].items():
        if sha256_file(ROOT/name)!=digest:
            raise ValueError(f"Changed follow-up source: {name}")
    for name,digest in receipt["artifacts"].items():
        if sha256_file(output/name)!=digest:
            raise ValueError(f"Changed follow-up artifact: {name}")
    allele=json.loads((output/"allele_manifest.json").read_text())
    if sha256_file(ROOT/"src/native_alleles.py")!=allele["parser_sha256"] or sha256_file(ROOT/"tools/extract_ctcf_alleles.py")!=allele["runner_sha256"]:
        raise ValueError("Allele parser or runner changed")
    for source in allele["sources"]:
        if sha256_file(ROOT/source["path"])!=source["sha256"]:
            raise ValueError("Published supplementary source changed")
    if sha256_file(output/"ctcf_allele_effects.csv")!=allele["table_sha256"]:
        raise ValueError("Allele effects changed")
    summary=json.loads((output/"summary.json").read_text())
    influence=pd.read_csv(output/"pilot_influence.csv")
    geometry=pd.read_csv(output/"geometry.csv")
    eligibility=pd.read_csv(output/"pilot_eligibility.csv")
    sequences=pd.read_csv(DEFAULT_CONFIG.paths.ctcf_dir/"ctcf_gm12878_sequences.tsv",sep="\t")
    if sha256_file(DEFAULT_CONFIG.paths.ctcf_dir/"ctcf_gm12878_sequences.tsv")!=json.loads((output/"protocol.json").read_text())["input_sha256"]:
        raise ValueError("Prepared CTCF source changed")
    lookup={f"{row.chrom}:{row.start}-{row.end}":row.sequence for row in sequences.itertuples()}
    pssm=motif_pssm(load_jaspar_ctcf_motif())
    # Continuous max scores include excluded sequences with no threshold-passing hit.
    eligibility["maximum_pwm_score"]= [float(np.nanmax(scan_sequence_with_pssm(lookup[name],pssm))) for name in eligibility.sequence_id]
    eligibility.to_csv(output/"eligibility_metrics.csv",index=False)
    eligibility.groupby("retained").agg(sequences=("sequence_id","size"),
        mean_length=("nucleotide_length","mean"),mean_gc=("gc_fraction","mean"),
        mean_pwm=("maximum_pwm_score","mean"),mean_tokens=("real_tokens","mean"),
        mean_token_width=("mean_token_width","mean")).to_csv(output/"eligibility_balance.csv")
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    ordered=influence.sort_values("motif_minus_sham_loss")
    axes[0,0].scatter(ordered.motif_minus_sham_loss,range(len(ordered)),c="#286e82")
    axes[0,0].set_yticks(range(len(ordered)),[name.replace("-",":").split(":")[0]+":"+name.split(":")[1].split("-")[0] for name in ordered.sequence_id],fontsize=7)
    axes[0,0].axvline(0,color="grey",linewidth=.8)
    axes[0,0].set(xlabel="Motif-minus-sham loss (nats)",title="Original pilot: individual contrasts")
    loo=influence.drop_duplicates("cluster").sort_values("leave_cluster_out_mean")
    axes[0,1].scatter(loo.leave_cluster_out_mean,range(len(loo)),c="#286e82")
    axes[0,1].axvline(0,color="grey",linewidth=.8)
    axes[0,1].axvline(influence.motif_minus_sham_loss.mean(),color="#b76e44",linestyle="--",label="All sequences")
    axes[0,1].set(xlabel="Mean after removing one entire cluster",ylabel="Removed cluster",title="Cluster influence")
    axes[0,1].legend(frameon=False,fontsize=8)
    for index,row in enumerate(geometry.itertuples()):
        axes[1,0].plot([0,1],[row.motif_gap_bp,row.sham_gap_bp],color="grey",alpha=.5)
    axes[1,0].set_xticks([0,1],["Motif","Sham"])
    axes[1,0].set(ylabel="Window-to-target gap (bp)",title="Original control proximity")
    for retained,label,color in ((False,"Excluded","#b76e44"),(True,"Retained","#286e82")):
        group=eligibility[eligibility.retained==retained]
        axes[1,1].scatter(group.gc_fraction,group.maximum_pwm_score,label=label,c=color)
    axes[1,1].set(xlabel="GC fraction",ylabel="Maximum continuous PWM score",title="Scanned eligibility population")
    axes[1,1].legend(frameon=False,fontsize=8)
    fig.savefig(output/"pilot_diagnostics.png",dpi=180)
    plt.close(fig)
    recovery=pd.read_csv(output/"recovery_scores.csv")
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout="constrained")
    for name,group in recovery.groupby("width_bin"):
        axes[0].scatter(group.token_width,group["rank"],alpha=.55,label=name)
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Masked token width (bp)",ylabel="Correct-token vocabulary rank",title="Independent recovery calibration")
    grouped=recovery.groupby("width_bin").agg(native=("log_probability","mean"),baseline=("frequency_baseline_log_probability","mean"))
    grouped.plot.bar(ax=axes[1],color=["#286e82","#b76e44"],rot=0)
    axes[1].set(xlabel="Token width (bp)",ylabel="Mean target log probability (nats)",title="Native model versus frequency baseline")
    axes[1].legend(["Native MLM","Frequency baseline"],frameon=False,fontsize=8)
    fig.savefig(output/"recovery_calibration.png",dpi=180)
    plt.close(fig)
    paragraphs=(
        "\\paragraph{Endpoint and control diagnostics.}\n"
        "A separate exploratory follow-up holds the original motif edit and target fixed and requires a same-side sham "
        "within two bases of the motif-to-target distance, preserving window width and token geometry. "
        "Two disjoint 19-base windows on the same side must differ in target distance by at least 19 bases. "
        f"Thus none of the {summary['geometry_evaluated']} original pairs has a feasible tightly matched same-sequence control. "
        "We record this exclusion without relaxing the tolerance or claiming a revised motif effect. "
        f"Removing one entire genomic cluster moves the original mean contrast from ${summary['leave_cluster_out_min']:.4f}$ "
        f"to ${summary['leave_cluster_out_max']:.4f}$ nats, including a sign change. Eligibility metrics cover all "
        f"{summary['pilot_scanned']} scanned sequences, including the {summary['pilot_retained']} retained examples.\n\n"
        "\\paragraph{Native recovery and intervention controls.}\n"
        f"A fixed population of {summary['calibration_sequences']} windows on chromosomes 16/17 masks one seeded random real token "
        "per window, without using model outcomes for selection. Token frequencies are estimated separately on chromosomes 14/15. "
        f"Median correct-token rank is {summary['median_rank']:.1f}, top-1 recovery is {100*summary['top1_rate']:.2f}\\%, "
        f"and mean target log probability is {summary['mean_log_probability']:.4f} nats versus "
        f"{summary['frequency_baseline_log_probability']:.4f} for the smoothed frequency baseline. These descriptive results "
        "show recovery utility on that population, without establishing motif-specific endpoint sensitivity. "
        "Identity context patches, complete final-residual rescue and reverse controls pass in all 24 motif/sham cases. "
        "Final-residual replacement includes the residual bypass; attention-only recovery is recorded separately. "
        "An engineered motif-dependent predictor also passes a known-mechanism rescue check through the same hook interface.\n\n"
        "\\paragraph{An independently measured allele benchmark is recoverable.}\n"
        "The published supplemental Table S3 for GSE81945 supplies 16 mutation rows with pooled WGS and CTCF ChIP counts \\citep{poulos2016ctcf}. "
        "Its reference alleles agree with UCSC hg19 at the listed one-based positions. "
        f"There are {allele['heterozygous_variants']} heterozygous variants in {allele['heterozygous_loci']} locus groups; "
        "one homozygous row cannot identify a normalized allele effect. Two groups contain adjacent variants with unresolved phase. "
        "Input-normalized effects and descriptive half-count log-odds intervals are saved, but pooled counts cannot recover "
        "replicate variation or mapping-bias uncertainty. This small selected benchmark lacks substitution-matched controls "
        "and does not supply fresh biological confirmation.\n"
    )
    path=ROOT/"paper/native_followup.tex"
    path.write_text(paragraphs,encoding="utf-8",newline="\n")
    write_json(output/"report_manifest.json",dict(generator_sha256=sha256_file(Path(__file__)),
        execution_sha256=sha256_file(output/"execution.json"),allele_manifest_sha256=sha256_file(output/"allele_manifest.json"),
        paper_tex_sha256=sha256_file(path),outputs={p.name:sha256_file(p) for p in [output/"eligibility_metrics.csv",output/"eligibility_balance.csv",output/"pilot_diagnostics.png",output/"recovery_calibration.png"]}))
    print("verified follow-up evidence and generated manuscript diagnostics")


if __name__=="__main__":
    build()
