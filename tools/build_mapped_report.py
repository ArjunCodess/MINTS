"""Audit evidence and generate diagnostic tables, figures and manuscript insert."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from src.utils import write_json,sha256_file
from src.variant_statistics import cluster_summary
from tools.audit_mapped_variant import audit

ROOT=Path(__file__).resolve().parents[1]


def build():
    audit()
    output=ROOT/"results/mapped_variant/report"
    output.mkdir(parents=True,exist_ok=True)
    s=json.loads((ROOT/"results/mapped_variant/summary.json").read_text())
    d=json.loads((ROOT/"results/control_diagnostics/summary.json").read_text())
    cases=pd.read_csv(ROOT/"results/control_diagnostics/cases.csv")
    predicates=pd.read_csv(ROOT/"results/control_diagnostics/candidate_predicates.csv.gz")
    f=pd.read_csv(ROOT/"results/mapped_variant/case_scores.csv")
    g=pd.read_csv(ROOT/"results/mapped_variant/cluster_scores.csv")
    sources=pd.read_csv(ROOT/"results/adastra_exploratory/candidates.csv.gz")
    members=pd.read_csv(ROOT/"results/control_diagnostics/membership.csv")
    selected_keys=["original","local_only","local_no_width","local_geometry32","local_geometry64"]
    rows=[]
    for cohort,group in cases.groupby("cohort"):
        for k in ("full_allele_offsets","reference_motif","query_correspondence",*selected_keys):
            rows.append(dict(cohort=cohort,constraint=k,passing=int(group[k].sum()),denominator=len(group)))
    pd.DataFrame(rows).to_csv(output/"case_constraints.csv",index=False,lineterminator="\n")
    predicates=predicates.merge(cases[["variant_id","reference_motif","full_allele_offsets"]],on="variant_id",validate="many_to_one")
    keys=["outside_motif","edit_width","query_side","geometry16","gc","motif_locations","motif_score","full_sham_offsets","query_identity"]
    independent=[]
    for cohort,group in predicates.groupby("cohort"):
        for population,sub in (("all exact-trinucleotide candidates",group),("motif-overlapping variants",group[group.reference_motif])):
            for k in keys:
                independent.append(dict(cohort=cohort,risk_set=population,predicate=k,passing=int(sub[k].sum()),denominator=len(sub)))
    pd.DataFrame(independent).to_csv(output/"candidate_constraints.csv",index=False,lineterminator="\n")
    risk=predicates[predicates.reference_motif].copy()
    risk["failed_predicates"]=risk.apply(lambda r:"|".join(k for k in keys if not r[k]) or "none",axis=1)
    risk.groupby(["cohort","failed_predicates"]).size().rename("candidates").reset_index().to_csv(
        output/"constraint_intersections.csv",index=False,lineterminator="\n")
    base=sources.assign(substitution=sources.reference+">"+sources.alternate)
    comparisons=[]
    for cohort,sub in members.groupby("cohort"):
        sub=sub.assign(substitution=sub.reference+">"+sub.alternate)
        for key in ("chrom","substitution"):
            a=base[key].value_counts(normalize=True);b=sub[key].value_counts(normalize=True).reindex(a.index,fill_value=0)
            comparisons.append(dict(cohort=cohort,variable=key,total_variation=float(abs(a-b).sum()/2)))
    write_json(output/"representativity.json",dict(comparisons=comparisons,
        limitation="descriptive coordinate/allele distributions; does not certify motif/sequence or biological representativeness",
        original_zero_upper_approximation=3/4096,expansion_strict_retained=1,expansion_denominator=8192))
    stable=f[f.full_boundary_stable]
    stable_summary=cluster_summary(stable.contrast,stable.cluster,repetitions=10000)
    implementation=json.loads((ROOT/"results/mapped_variant/scores.json").read_text())
    errors=[r[k] for c in implementation for r in c["query_diagnostics"] for k in
            ("identity_errors","final_restoration_errors","repeat_errors")]
    diagnostics=dict(full_boundary_stable_summary=stable_summary,
        status="post-inspection descriptive summary of a prespecified tokenization stratum; no additional significance test",
        equal_cluster_variant_js=float(g.variant.mean()),equal_cluster_sham_js=float(g.sham.mean()),
        query_measurements=sum(len(c["query_diagnostics"]) for c in implementation),
        max_implementation_logit_error=max(v for values in errors for v in values),
        max_normalization_error=max(r["normalization_error"] for c in implementation for r in c["query_diagnostics"]),
        negative_case_contrasts=int((f.contrast<0).sum()),leave_cluster_out_min=min(x["mean"] for x in s["leave_cluster_out"]),
        leave_cluster_out_max=max(x["mean"] for x in s["leave_cluster_out"]))
    rng=np.random.default_rng(1731)
    correlations=[]
    for _ in range(10000):
        draw=rng.integers(0,len(g),len(g))
        x,y=g.variant.to_numpy()[draw],g.absolute_pwm.to_numpy()[draw]
        if np.ptp(x)>0 and np.ptp(y)>0:
            correlations.append(float(spearmanr(x,y).statistic))
    diagnostics["pwm_rho_interval"]=list(map(float,np.quantile(correlations,[.025,.975])))
    diagnostics["pwm_rho_defined_draws"]=len(correlations)
    diagnostics["pwm_interval_status"]="post-score uncertainty calculation for the frozen secondary endpoint; primary gate unchanged"
    write_json(output/"diagnostics.json",diagnostics)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    x=np.arange(5);labels=["original","mapped query","+ no width","+ 32 bp","+ 64 bp"]
    for cohort,offset,color in (("original4096",-.18,"#366b84"),("expansion",.18,"#bd713b")):
        count=[d["cohorts"][cohort][k] for k in selected_keys]
        denom=d["cohorts"][cohort]["records"]
        ax[0].bar(x+offset,np.array(count)/denom*100,width=.36,label=f"{cohort}, n={denom}",color=color)
    ax[0].set_xticks(x,labels,rotation=25,ha="right");ax[0].set_ylabel("sequence eligibility (%)")
    ax[0].legend(frameon=False,fontsize=8);ax[0].set_title("Exploratory diagnostic alternatives")
    sorted_values=np.sort(g.contrast.to_numpy())
    ax[1].plot(sorted_values,np.arange(len(g)),"o",markersize=3,color="#366b84")
    ax[1].axvline(0,color="black",linewidth=.8);ax[1].set_xlabel("variant minus sham JS (nats)")
    ax[1].set_ylabel("genomic cluster, sorted");ax[1].set_title("Frozen mapped-query study")
    fig.savefig(output/"study.png",dpi=180);plt.close(fig)
    primary=s["primary"]
    table="\n".join(f"| {label} | {s[key]:,} |" for label,key in (
        ("Source records","source_records"),("Selected records","selected"),("Reference verified","reference_verified"),
        ("Motif eligible","motif_eligible"),("Query correspondence eligible","correspondence_eligible"),
        ("Matched controls","matched_controls"),("Implementation-valid cases","implementation_valid"),
        ("Genomic proxy clusters","genomic_proxy_clusters")))
    provenance=json.loads((ROOT/"results/variant_provenance/provenance_summary.json").read_text())
    report=f"""# Mapped-query exploratory study

The separately frozen study completed with a negative variant-minus-sham native
sensitivity contrast. All {s['implementation_valid']} cases passed implementation checks,
but the sensitivity gate failed. Head search and confirmation did not run. This
is an interpretable endpoint for the declared study, not completion of a CTCF
mechanism claim.

## Control diagnosis

The expanded hash screen selected 12,288 of 512,556 coverage-eligible archived
CTCF records, retaining nonsignificant records. Every selected hg38 reference
allele verified. Reference motifs overlapped 160/4,096 original cases and 273/8,192
expansion cases. Complete reference/alternate BPE correspondence held in 1,627
and 3,223 respectively. Identical local query spans and vocabulary IDs existed
for all 12,288, so complete input correspondence excluded many usable query tasks.
These are independently evaluated quantities, not a decomposition of sequential
first-rejection rates.

Same-side query distance differences equal edit separation. The 16-base tolerance
therefore shrinks the nominal 64-base sham search to 16 bases; requiring an identical
trinucleotide, motif exclusion, width, GC and stable tokenization intersects that
small set. This is excessive attrition, not universal single-SNV incompatibility.
The original rules retain 0/4,096 and 1/8,192. The expanded screen confirms that
strict controls are rare; it does not justify saying they never exist. The rough
rule-of-three upper rate for zero in the original screen is 0.073%, conditional
on treating the hash sample as approximately random. Coordinate and substitution
frequency comparisons are saved in `report/representativity.json`; they do not
certify biological representativeness.

Identical query correspondence alone retains 2 and 9 cases. Removing reference
edit-token-width matching as well retains 21 and 40. Allowing distance differences
of 32 or 64 bases would retain 108 or 176 overall. Those alternatives were logged
as sequence-only method-development attempts and were not scored. The adopted
design changes only complete BPE and width matching, preserving tighter nucleotide
geometry, substitution, trinucleotide, GC and motif controls. Anchored masking,
between-window controls and a different model were not adopted because this
minimal change supplies a viable assay without changing model or query targets.
The effects of segmentation outside queries remain a limitation.

A separate sequence-only supplement diagnoses the 11 singleton heterozygous
GSE81945 loci using their preserved hg19 windows. All 11 have reference motifs and
shared local query spans, but only four have complete allele BPE correspondence.
The original rules still retain zero; query correspondence alone would retain one,
and removing width matching would retain two. These post-native exploratory
diagnostics neither reopen the stopped study nor authorize scoring GSE81945.
The single strict ADASTRA expansion case was independently rechecked with the
original sequence-control function and retained the same sham at index 113.

## Population and implementation

| Denominator | Records |
| --- | ---: |
{table}
| Biologically QC eligible | unknown |
| Independent donors | unknown |

The 61 cases are reference-plus-one-SNV scenarios, not verified individual
haplotypes. Query prediction tasks match exact half-open nucleotide spans and
vocabulary identities across each triplet. Different tensor indices are mapped
explicitly; 29 cases change at least one query index. Masking leaves the intended
SNV and sham visible. Each query is masked separately and case aggregation weights
its nucleotide width. The pinned model loaded its native prediction head with no
missing weights; model/tokenizer/cache hashes and versions are saved in `model.json`.

Engineered fixtures verify channel-specific mapped head transfer across different
input lengths and indices. Natural-case identity, repeated execution, final-query
reference restoration and reverse restoration pass the 0.0002 maximum-logit-error
tolerance. Observed maximum error is {diagnostics['max_implementation_logit_error']:.8g}.
Final-query restoration must recover reference logits because the native prediction
head is pointwise. A single head-context intervention has no such expectation.
These checks establish wiring, not sensitivity to natural motif effects.

## Native findings

Equal-cluster mean variant JS is {diagnostics['equal_cluster_variant_js']:.6f} nats,
versus {diagnostics['equal_cluster_sham_js']:.6f} for shams. The primary contrast is
{primary['mean']:.6f} nats, with a 95% cluster-bootstrap interval of
[{primary['ci_low']:.6f}, {primary['ci_high']:.6f}]. Native predictions respond to
both perturbations, but the matched motif-preserving shams produce larger changes.
The result contradicts the prespecified positive differential-sensitivity hypothesis
on this retained population. It does not show absent variant sensitivity or absent
biological variant effects.

Every leave-one-cluster-out mean remains negative, ranging from
{diagnostics['leave_cluster_out_min']:.6f} to {diagnostics['leave_cluster_out_max']:.6f}.
Chromosome aggregation gives {s['chromosome_sensitivity']['mean']:.6f} with interval
[{s['chromosome_sensitivity']['ci_low']:.6f}, {s['chromosome_sensitivity']['ci_high']:.6f}].
These are dependence-aware summaries over genomic proxies, not verified independent
donors. Shared source experiments are irrelevant to sequence-only computations
but would matter for biological association.

The prespecified width-matched subset contains 11 cases/clusters, with mean
{s['width_matched']['mean']:.6f} and interval
[{s['width_matched']['ci_low']:.6f}, {s['width_matched']['ci_high']:.6f}]. Its Holm-adjusted
secondary p-value is {s['secondary_tests'][1]['p_holm']:.4f}. Absolute motif-score change
and variant JS have cluster-level Spearman rho {s['secondary_tests'][0]['effect']:.4f},
Holm p={s['secondary_tests'][0]['p_holm']:.4f}, on identical primary membership. Neither
secondary test supplies positive evidence. P-values use the declared cluster
summaries; rank-test asymptotics and the width-subset t-test assume independent
proxy clusters and are exploratory.
The secondary rho interval is [{diagnostics['pwm_rho_interval'][0]:.4f},
{diagnostics['pwm_rho_interval'][1]:.4f}], from {len(correlations)} defined cluster draws.
This uncertainty calculation was added during reporting for the frozen secondary
endpoint; it does not change membership, scoring, tests or the primary decision.

Fifteen triplets have complete stable boundaries, enough for the protocol's
tokenization-isolation *count* gate. Their post-inspection descriptive contrast is
{stable_summary['mean']:.6f}, interval [{stable_summary['ci_low']:.6f}, {stable_summary['ci_high']:.6f}].
All 15 case contrasts are negative. This stratum does not restore edit-token-width
matching or create a new confirmatory test. Full-population differences may reflect
segmentation, ALiBi position, sequence context, generic perturbation sensitivity or
selection of unusually matchable motifs. The study cannot distinguish these fully.

## Biological provenance and stopping

The publisher readme was downloaded and MD5-verified. It defines archived effects
as weighted log2 observed/expected allele ratios, not GSE81945 input-normalized
log odds. Experiment metadata contain 685 CTCF experiments; the live API exposed
238 distinct CTCF ChIP count observations for 26 of the 61 cases, spanning 174
experiments and 36 source series. All 238 link to experiment metadata, but these
observations connect into only five components through shared experiments.
They are not 238 independent donors. The live v6.1.1 provenance is a separately
inspected supplement, not replacement of the frozen v6.1 archive.

Of 61 official SNP-detail requests, 35 returned coordinate-verified records and
26 returned HTTP 404. Nine returned records lacked exposed CTCF observations.
The documented official rsID search was also tried for all 26 missing detail
records; all returned HTTP 200 with zero indexed results. This alternative did
not recover the missing experiment-level measurements.
An absent API observation means unknown availability, not zero coverage or an
absent effect. Request receipts and actual available counts are preserved.
Individual genotypes/phase, source-specific donor identities, per-replicate input
counts and mapping-corrected alignments remain unverified, so no biological
correlation is computed and QC-eligible case/donor counts remain null.

ENCODE metadata identify three GM12878 ChIP replicates sharing one biosample and
one donor, with a linked input experiment. This previously inspected assay is
not fresh confirmation. Its compressed ChIP/input reads total 2.340 GB. Raw
processing was not started with only about 8 GB spare disk: a conservative 30 GiB
working floor covers reference/index, read remapping and BAM/sort stages. That is
a planning estimate, not measured allocation. WASP allele-swap/remap is documented
as the required mapping correction; no unprocessed reads are presented as QC.
GEO SOFT resolves GSE81945 to PRJNA323481/SRP075768, with 5.208 GB of listed FASTQ
data. An initial candidate ENA query used PRJNA324773; it is retained as unassigned
metadata and excluded from GSE81945 accounting.

Feasibility, implementation and the boundary-stable count gate pass. Native
differential sensitivity and biological QC fail, stopping causal head discovery,
intervention-based power and independent confirmation. No selected maximum, causal
rescue, biological agreement or independent replication is claimed. A symmetric JS
endpoint cannot identify which allele binds more strongly. The earlier native
masked-flank sensitivity failure and GSE81945 zero-control result remain intact.

## Reproduction and evidence

Run `python tools/run_control_diagnostics.py --output NEW_DIAGNOSTIC` for a fresh
diagnostic. The frozen native runner consumes the saved diagnostic membership;
run `python tools/run_mapped_variant.py --output NEW_STUDY` with pinned local
model/tokenizer and hg38. The default raw-logit directory is attempt-specific;
use `--raw-directory PATH` for repeated runs. Both runners refuse existing output
directories. The recorded RTX 4060 float32 native run took about 60 seconds after
the diagnostic, which takes minutes; raw logits and downloads remain under ignored
`data/adastra/`. Compact cases, query maps, per-query scores, cluster scores,
snapshotted source and execution hashes are committed.

`python tools/audit_mapped_variant.py` needs no scientific cache or network;
`--raw` also verifies local raw-logit checksums and recomputes every JS value.
`python tools/build_mapped_report.py` audits evidence and regenerates this report,
tables, figure and manuscript insert. The completed endpoint is a bounded negative
exploratory sequence result, with a separate biological provenance limit.
"""
    (ROOT/"docs/mapped_variant_results.md").write_text(report,encoding="utf-8",newline="\n")
    latex=f"""% Generated by tools/build_mapped_report.py from audited frozen evidence.
\\subsection{{Expanded natural-variant diagnostic and mapped-query study}}
The archived ADASTRA Mabel v6.1 CTCF table \\citep{{adastra2024release,abramov2021adastra}} contains 512,556 coverage-eligible
records, including nonsignificant records. A prior 4,096-case hash screen retained
no strict controls. A separately frozen exploratory expansion to 12,288 verified
hg38 single-SNV reference scenarios retained one strict control in the added 8,192
cases. Complete allele BPE offsets agree in 4,850 cases; exact local query spans
and token identities exist in all 12,288. Reference motifs cover 433 variants.
Same-side query distance differences equal edit separation, making the 16-base
tolerance more restrictive than the nominal 64-base sham search. Independent
candidate predicates and their intersections diagnose attrition without changing
the original sequential eligibility decisions.

A new exploratory protocol removes complete input-offset and edit-token-width
matching while retaining exact query correspondence, substitution, trinucleotide,
GC, motif preservation and 16-base geometry controls. It retains 61 cases in 60
genomic/sequence proxy clusters. Query indices are explicitly mapped, and 29 cases
change an index. Each unchanged query is masked separately with both edits visible;
Jensen--Shannon divergence is weighted by nucleotide width. Native identity, repeat,
mapped final-query restoration and reverse restoration pass implementation checks.
The pointwise MLM head makes final-query representation restoration an exact
logit-recovery control; no such requirement is imposed on individual heads.

Variant and sham mean divergences are {diagnostics['equal_cluster_variant_js']:.6f}
and {diagnostics['equal_cluster_sham_js']:.6f} nats. The equal-cluster contrast is
{primary['mean']:.6f}, with 95\\% bootstrap interval
[{primary['ci_low']:.6f}, {primary['ci_high']:.6f}]. The shams perturb predictions more,
failing the positive differential-sensitivity gate. Every leave-cluster-out mean
remains negative; chromosome aggregation also remains negative. The width-matched
11-cluster subset has contrast {s['width_matched']['mean']:.6f} with interval
[{s['width_matched']['ci_low']:.6f}, {s['width_matched']['ci_high']:.6f}]. Its Holm p-value
is {s['secondary_tests'][1]['p_holm']:.4f}; absolute PWM-change agreement has Spearman
rho {s['secondary_tests'][0]['effect']:.4f}, Holm p={s['secondary_tests'][0]['p_holm']:.4f}.
Fifteen boundary-stable triplets all have negative contrasts, a descriptive
post-inspection diagnostic rather than a new confirmation test.

This result concerns generic native prediction sensitivity on a small matchable
population. It does not identify binding direction, absent variant sensitivity,
or a causal CTCF mechanism. Segmentation, positional effects, sequence context,
indirect binding and selected-population effects remain plausible alternatives.
ADASTRA weighted log2 observed/expected effects are not interchangeable with
GSE81945 input-normalized log odds. Live provenance retrieval exposes 238 ChIP
allele-count observations for 26 cases across 174 experiments, but only five
components after joining shared experiments. Donor/phase, per-replicate input,
mapping correction and dosage uncertainty remain unverified. Biological QC
therefore fails separately from native sensitivity. No biological correlation,
head discovery, intervention-based power or independent confirmation is run.
Earlier stopped protocols and results remain intact.

\\begin{{figure}}[t]
\\centering\\includegraphics[width=\\linewidth]{{../results/mapped_variant/report/study.png}}
\\caption{{Sequence-only diagnostic retention under declared alternatives and
cluster-level native contrasts under the separately frozen mapped-query design.
Alternatives were examined before scoring and are exploratory. Genomic clusters
are dependence proxies, not verified independent donors.}}
\\end{{figure}}
"""
    (ROOT/"paper/mapped_variant.tex").write_text(latex,encoding="utf-8",newline="\n")
    inputs=[ROOT/"results/mapped_variant/summary.json",ROOT/"results/mapped_variant/scores.json",
        ROOT/"results/mapped_variant/case_scores.csv",ROOT/"results/control_diagnostics/summary.json",
        ROOT/"results/variant_provenance/provenance_summary.json"]
    artifacts=[p for p in output.iterdir() if p.is_file()]+[ROOT/"docs/mapped_variant_results.md",ROOT/"paper/mapped_variant.tex"]
    write_json(ROOT/"results/mapped_variant/report_manifest.json",dict(builder_sha256=sha256_file(Path(__file__)),
        inputs={p.relative_to(ROOT).as_posix():sha256_file(p) for p in inputs},
        outputs={p.relative_to(ROOT).as_posix():sha256_file(p) for p in artifacts}))
    print("mapped report, tables, figure and manuscript insert built")


if __name__=="__main__":build()
