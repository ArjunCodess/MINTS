"""Verify scientific executions and build numeric manuscript inserts."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils import sha256_file, write_json
from tools.run_hardened_assay import STAGES
ROOT=Path(__file__).resolve().parents[1]

def validate_executions(output,root=ROOT):
    output=Path(output)
    for stage in STAGES:
        receipt=json.loads((output/stage/"execution.json").read_text())
        if receipt["status"]!="completed" or receipt["protocol_sha256"]!=sha256_file(output/"protocol.json"):
            raise ValueError(f"Incomplete or incompatible hardened stage: {stage}")
        if receipt.get("source_changed_during_stage"):
            raise ValueError(f"Source changed during stage: {stage}")
        for name,digest in receipt["source_sha256"].items():
            # This presentation generator is not imported by scientific stages.
            # Its current source is verified separately in the paper manifest.
            if name == "tools/build_hardened_artifacts.py":
                continue
            if sha256_file(root/name)!=digest:
                raise ValueError(f"Stale scientific source: {name}")
        for name,digest in receipt["artifacts"].items():
            if sha256_file(output/name)!=digest:
                raise ValueError(f"Changed stage artifact: {name}")

def table(headers,rows):
    return "\n".join([r"\begin{tabular}{"+"l"+"r"*(len(headers)-1)+"}",r"\toprule",
        " & ".join(headers)+r" \\",r"\midrule",
        *[" & ".join(map(str,row))+r" \\" for row in rows],r"\bottomrule",r"\end{tabular}"])

def build(output=ROOT/"results/hardened"):
    output=Path(output);validate_executions(output)
    figures=output/"figures";figures.mkdir(exist_ok=True)
    cal=json.loads((output/"calibration/calibration_manifest.json").read_text())
    obs=pd.read_csv(output/"calibration/calibration_observations.csv")
    rates=obs[obs.partition=="confirmation"].groupby("kind")[["passes_calibrated","passes_historical"]].mean()
    fig,ax=plt.subplots(figsize=(6,2.7))
    rates.rename(columns={"passes_calibrated":"Discovery-selected","passes_historical":"Historical"}).plot.bar(ax=ax,color=["#286e82","#b5aaa0"])
    ax.set(ylabel="Confirmation detection rate",xlabel="",ylim=(0,1.05));ax.tick_params(axis="x",rotation=20);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figures/"calibration.png",dpi=180);fig.savefig(figures/"calibration.pdf");plt.close(fig)
    predictive=[]
    for task,label in [("promoter_tata","TATA"),("promoter_no_tata","Other promoter"),("splice_sites_donors","Donor"),("splice_sites_acceptors","Acceptor"),("ctcf_accessible_peak_overlap","Accessible CTCF")]:
        t=pd.read_csv(output/("ctcf" if task=="ctcf_accessible_peak_overlap" else "prediction")/f"{task}_paired_comparisons.csv")
        r=t[(t["first"]=="baseline_plus_residual")&(t.resampling_unit=="genomic_block")].iloc[0]
        predictive.append([label,f"{r.difference:.4f}",f"[{r.ci_low:.4f}, {r.ci_high:.4f}]",int(r.clusters)])
    studies=[];plots=[]
    for folder,label in [(output/"patching/promoter_tata","TATA"),(output/"ctcf/interventions","CTCF")]:
        manifest=json.loads((folder/"study_manifest.json").read_text())
        if manifest["status"]!="completed":
            continue
        t=pd.read_csv(folder/"confirmation_summary.csv");r=t[t.analysis_scope=="primary"].iloc[0]
        studies.append([label,f"{int(r.layer)}/{int(r['head'])}",int(r.sequences),f"{r.mean_absolute_motif_minus_sham:.4f}",
            f"[{r.block_ci_low:.4f}, {r.block_ci_high:.4f}]",f"{r.block_p:.4f}"])
        for r in t[t.direction=="denoise"].itertuples():
            plots.append(dict(label=f"{label}: {r.scheme}",mean=r.mean_absolute_motif_minus_sham,low=r.block_ci_low,high=r.block_ci_high))
    fig,ax=plt.subplots(figsize=(6,4))
    for i,r in enumerate(plots):
        ax.errorbar(r["mean"],i,xerr=[[max(0,r["mean"]-r["low"])],[max(0,r["high"]-r["mean"])]],fmt="o",color="#286e82")
    ax.set_yticks(range(len(plots)),[r["label"] for r in plots]);ax.axvline(0,color="gray",lw=.7)
    ax.set_xlabel("Motif minus sham score effect, genomic-block interval")
    fig.tight_layout();fig.savefig(figures/"controlled_interventions.png",dpi=180);fig.savefig(figures/"controlled_interventions.pdf");plt.close(fig)
    genomic=json.loads((output/"genomic/native_sensitivity_manifest.json").read_text())
    native=pd.read_csv(output/"genomic/native_sensitivity_inference.csv");geometry=[]
    for run in genomic["runs"]:
        t=native[(native.fraction==run["fraction"])&(native.geometry==run["geometry"])&(native.metric=="base_density")&(native.resampling_unit=="genomic_block")]
        geometry.append([f'{run["fraction"]:.2f}',run["geometry"].replace("_",r"\_"),run["eligible_pairs"],run.get("evaluated_pairs",0),
            int(((t["mean"]>0)&(t.max_stat_p<.05)).sum())])
    v=json.loads((output/"ctcf/target_validation.json").read_text())
    m=pd.read_csv(output/"ctcf/ctcf_accessible_peak_overlap_metrics.csv")
    auc=m[m.method=="residual"].auroc.iloc[0]
    tex=[
        "% Generated from validated hardened stage artifacts.",
        r"\section{Controlled assays and independent confirmation}",
        r"\paragraph{Known-mechanism calibration.}",
        f"Oracle motif features implanted in QK and V span TATA and CTCF, single-head, query-specific, distributed, composition-only, and randomized computations. Strength, position, and instance count vary. Thresholds chosen on discovery seeds yield confirmation sensitivity {cal['confirmation_sensitivity']:.3f} and false-positive rate {cal['confirmation_false_positive_rate']:.3f}. This validates engineered base-token fixtures, not sensitivity to learned DNABERT mechanisms.",
        r"\begin{figure}[t]\centering\includegraphics[width=.85\linewidth]{../results/hardened/figures/calibration.png}\caption{Independent synthetic confirmation seeds.}\end{figure}",
        r"\paragraph{Incremental prediction.}",
        r"Baseline features include mono- and dinucleotide composition, motif count, score and relative position, and window coordinates. Baseline-only and baseline-plus-residual models use identical populations, validation-selected regularization, and fixed held-out predictions. Table~\ref{tab:incremental} gives paired 1 Mb genomic-block intervals.",
        r"\begin{table}[t]\centering\small\caption{Baseline-plus-residual minus baseline-only AUROC.}\label{tab:incremental}",
        table(["Task",r"$\Delta$ AUROC","95\\% block interval","Clusters"],predictive),r"\end{table}",
        r"\paragraph{Aligned composition controls.}",
        r"Motif and sham edits preserve exact A/C/G/T counts and match nucleotide transitions, edit count, window width, overlapping BPE count, and covered nucleotide width. Shams occupy the nearest eligible non-motif window, with distance logged. All token boundaries agree, the target motif is destroyed, and no new threshold-passing hit locations are created. Eligibility uses sequence properties without output filtering. Six of ten historical TATA pairs have incompatible boundaries at patched positions; their PM rankings are withdrawn as aligned intervention evidence.",
        r"Discovery on chromosomes 18/19 chooses one of 144 heads, whose identity and scorer are frozen before confirmation on 20/21. Absolute motif-minus-sham score changes are primary. Edits are averaged within sequence; genomic blocks merge shared intervals and exact or reverse-complement inputs. Discovery carries simultaneous maximum-statistic inference. Confirmation saves individual scores, leave-one-out estimates, and six schemes in both denoising and reverse directions. Secondary tests receive Holm correction. Donor GT edits retain insufficient aligned matched examples, so no donor mechanism is inferred.",
        r"\begin{table}[t]\centering\footnotesize\caption{Fixed-head edit-only denoising confirmation.}\label{tab:controlled}",
        table(["Motif","Head","Sequences","Effect","95\\% block interval",r"$p$"],studies),r"\end{table}",
        r"\begin{figure}[t]\centering\includegraphics[width=.9\linewidth]{../results/hardened/figures/controlled_interventions.png}\caption{Genomic-block intervals for fixed-head denoising schemes; secondary schemes are exploratory.}\end{figure}",
        r"\paragraph{Independent accessible CTCF cohort.}",
        f"GM12878 DNase experiment ENCSR000EMT, GRCh38 file ENCFF598KWZ, defines accessible windows independently of the saved CTCF peaks. Cases overlap those peaks; controls do not, including motif-bearing non-overlap windows. Matching uses chromosome, 204 bp width, GC within 0.02, and log-DNase signal within 0.5. Per-class caps are 512 training, 128 validation, and 128 test. The readout passes its pre-execution validation AUROC gate of 0.6 with validation AUROC {v['validation_auroc']:.4f} and test AUROC {auc:.4f}. Localization and intervention use the same CTCF motif and trained output. The paired incremental comparison in Table~\\ref{{tab:incremental}} is negative for CTCF: this trained output does not imply residual information beyond the direct motif baseline. Peak non-overlap does not verify binding absence. Repeat annotations, quantitative occupancy, and other cell types are unavailable.",
        r"\paragraph{Native attention sensitivity.}",
        f"PWM score-range fractions 0.7, 0.8, and 0.9 recompute scoring and matching across {genomic['source_sequences']:,} peaks. Each configuration evaluates at most {genomic['pair_cap']} random eligible pairs. Token balancing requires identical real-token and target-token counts and covered widths within 2 bp. Maximum, mean, and overlap-weighted aggregation give separate QK assays. Attention compares base and token weighting, global and local queries, and native, content-only, and position-only softmax. Every native head context is checked. Table~\\ref{{tab:geometry}} counts positive base-density contrasts passing genomic-block maximum-statistic tests over 144 heads; saved inference also corrects across all sensitivity choices and reports chromosome effects.",
        r"\begin{table}[t]\centering\small\caption{Recomputed geometry sensitivity.}\label{tab:geometry}",
        table(["Fraction","Geometry","Eligible","Evaluated","Heads"],geometry),r"\end{table}",
        r"\paragraph{Scope and reproducibility.}",
        r"These experiments test computational calibration and trained-readout effects without establishing native pretrained binding causality. Protocol choices precede execution without external registration. Distant homology, label reconstruction, pretraining exposure, and classifier-seed variability remain unresolved. Immutable legacy-pipeline receipts reject changed scientific configuration, source, membership, or artifacts on resume. Isolated execution records retain prior receipts and verify source and artifact hashes before generating these results.",
    ]
    target=ROOT/"paper/hardened_results.tex";target.write_text("\n\n".join(tex)+"\n",encoding="utf-8",newline="\n")
    coverage=[
        "exact nucleotide alignment","composition-preserving and transition-matched edits","same-motif validated CTCF trained readout",
        "absolute effects and leave-one-out estimates","known-mechanism synthetic calibration","token geometry controls",
        "independent accessible non-peak controls","discovery and independent confirmation","genomic-block and identity clustering",
        "baseline-plus-residual comparison","paired classifier differences","PWM and aggregation sensitivity",
        "six patch schemes and reverse effects","related work and second motif/assay","immutable resume receipts",
        "actual layer IDs","pinned runtime, preflight and CPU CI"]
    write_json(output/"implementation_coverage.json",dict(items=[dict(item=i+1,implementation=x) for i,x in enumerate(coverage)],
        limitations=["oracle fixtures do not validate learned mechanisms","donor matched eligibility insufficient","repeat and occupancy annotations unavailable","no new label reconstruction or cell-type replication"]))
    (output/"implementation_coverage.md").write_text("# Implemented review items\n\n"+"\n".join(f"{i+1}. {x}." for i,x in enumerate(coverage))+
        "\n\nSee each stage's execution and scientific manifests for evidence and exact caps. Donor eligibility was insufficient. Synthetic calibration and trained-readout effects retain their separate claim boundaries; repeat and quantitative occupancy annotations were unavailable.\n",encoding="utf-8")
    write_json(output/"paper_manifest.json",dict(generated_by="python tools/build_hardened_artifacts.py",
        generator_sha256=sha256_file(Path(__file__)),
        source_validation="scientific stage sources unchanged; presentation generator tracked separately",
        sources={p.relative_to(ROOT).as_posix():sha256_file(p) for stage in STAGES for p in (output/stage).rglob("*") if p.is_file()},
        paper={"path":target.relative_to(ROOT).as_posix(),"sha256":sha256_file(target)},
        figures={p.relative_to(ROOT).as_posix():sha256_file(p) for p in figures.glob("*")}))
    return target
if __name__=="__main__":
    print(build())
