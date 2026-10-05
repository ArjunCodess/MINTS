"""Post-hoc descriptive balance audit; no model/component selection."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def complexity(sequence):
    text=str(sequence)
    p=np.asarray([text.count(base)/len(text) for base in "ACGT"])
    entropy=float(-(p[p>0]*np.log2(p[p>0])).sum())
    longest=run=1
    for first,second in zip(text,text[1:]):
        run=run+1 if first==second else 1
        longest=max(longest,run)
    return entropy,longest

def build():
    source=ROOT/"data/ctcf/ctcf_gm12878_sequences.tsv"
    peaks=pd.read_csv(source,sep="\t")
    summary=[];members=[];inputs=[source]
    for path in sorted((ROOT/"results/hardened/genomic").glob("*_pairs.csv")):
        pairs=pd.read_csv(path);inputs.append(path)
        for role in ["present","absent"]:
            selected=peaks.iloc[pairs[f"{role}_index"].to_numpy()]
            entropy,runs=zip(*(complexity(s) for s in selected.sequence))
            members.extend(dict(configuration=path.stem,role=role,name=str(row.name),
                bed_score=float(row.score),base_entropy=entropy[i],longest_mononucleotide_run=runs[i])
                for i,row in enumerate(selected.itertuples()))
            summary.append(dict(configuration=path.stem,role=role,n=len(selected),
                mean_bed_score=float(selected.score.mean()),median_bed_score=float(selected.score.median()),
                fraction_score_ceiling=float((selected.score==1000).mean()),
                mean_base_entropy=float(np.mean(entropy)),mean_longest_run=float(np.mean(runs))))
    pd.DataFrame(summary).to_csv(OUT/"native_peak_balance.csv",index=False)
    pd.DataFrame(members).to_csv(OUT/"native_peak_balance_members.csv",index=False)
    cohort_path=ROOT/"results/hardened/ctcf/ctcf_case_control.tsv"
    cohort=pd.read_csv(cohort_path,sep="\t");inputs.append(cohort_path)
    cohorts=[]
    for keys,group in cohort.groupby(["partition","label","motif_present"]):
        entropy,runs=zip(*(complexity(s) for s in group.sequence))
        cohorts.append(dict(partition=keys[0],label=int(keys[1]),motif_present=bool(keys[2]),n=len(group),
            mean_gc=float(group.gc.mean()),mean_log_dnase=float(np.log1p(group.accessibility_signal.clip(lower=0)).mean()),
            mean_base_entropy=float(np.mean(entropy)),mean_longest_run=float(np.mean(runs))))
    pd.DataFrame(cohorts).to_csv(OUT/"accessible_cohort_balance.csv",index=False)
    tex=[r"\paragraph{Additional descriptive balance audit.}",
        r"Normalized BED scores are available in the prepared CTCF peaks, although quantitative occupancy and repeat annotations are not. Table~\ref{tab:bed-balance} reports their means in evaluated native-control pairs. Matching does not constrain this score, so differences remain possible confounders. Most peaks reach the score ceiling of 1000. Saved diagnostics also report base entropy, longest mononucleotide runs, and DNase balance by motif presence and partition; these sequence-complexity measures do not identify repeat annotations. This post-hoc audit does not select heads or modify inference.",
        r"\begin{table}[t]\centering\small\caption{Normalized CTCF BED-score balance in evaluated pairs.}\label{tab:bed-balance}",
        r"\begin{tabular}{lrr}\toprule Configuration & Present mean & Absent mean \\\midrule"]
    frame=pd.DataFrame(summary)
    for name,group in frame.groupby("configuration"):
        role=group.set_index("role")
        tex.append(name.replace("_",r"\_")+f" & {role.loc['present','mean_bed_score']:.1f} & {role.loc['absent','mean_bed_score']:.1f}"+r" \\")
    tex.extend([r"\bottomrule\end{tabular}",r"\end{table}"])
    paper=ROOT/"paper/cohort_balance.tex";paper.write_text("\n".join(tex)+"\n",encoding="utf-8",newline="\n")
    outputs=[*OUT.glob("*.csv"),paper]
    manifest=dict(scope="post-hoc descriptive balance, no biological inference or selection",
        command="python results/hardened/cohort_diagnostics/build_balance.py",
        script_sha256=digest(__file__),
        inputs={p.relative_to(ROOT).as_posix():digest(p) for p in inputs},
        outputs={p.relative_to(ROOT).as_posix():digest(p) for p in outputs},
        limitations=["BED rank/normalized score is not calibrated occupancy","base entropy and run length are not repeat annotation"])
    (OUT/"report_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(paper)

if __name__=="__main__":
    build()
