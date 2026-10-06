# Mapped-query exploratory study

The separately frozen study completed with a negative variant-minus-sham native
sensitivity contrast. All 61 cases passed implementation checks,
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
| Source records | 512,556 |
| Selected records | 12,288 |
| Reference verified | 12,288 |
| Motif eligible | 433 |
| Query correspondence eligible | 433 |
| Matched controls | 61 |
| Implementation-valid cases | 61 |
| Genomic proxy clusters | 60 |
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
tolerance. Observed maximum error is 0.
Final-query restoration must recover reference logits because the native prediction
head is pointwise. A single head-context intervention has no such expectation.
These checks establish wiring, not sensitivity to natural motif effects.

## Native findings

Equal-cluster mean variant JS is 0.013053 nats,
versus 0.029018 for shams. The primary contrast is
-0.015966 nats, with a 95% cluster-bootstrap interval of
[-0.024454, -0.009146]. Native predictions respond to
both perturbations, but the matched motif-preserving shams produce larger changes.
The result contradicts the prespecified positive differential-sensitivity hypothesis
on this retained population. It does not show absent variant sensitivity or absent
biological variant effects.

Every leave-one-cluster-out mean remains negative, ranging from
-0.016587 to -0.013039.
Chromosome aggregation gives -0.016271 with interval
[-0.026127, -0.008025].
These are dependence-aware summaries over genomic proxies, not verified independent
donors. Shared source experiments are irrelevant to sequence-only computations
but would matter for biological association.

The prespecified width-matched subset contains 11 cases/clusters, with mean
-0.006758 and interval
[-0.014121, 0.000623]. Its Holm-adjusted
secondary p-value is 0.2443. Absolute motif-score change
and variant JS have cluster-level Spearman rho 0.1428,
Holm p=0.2763, on identical primary membership. Neither
secondary test supplies positive evidence. P-values use the declared cluster
summaries; rank-test asymptotics and the width-subset t-test assume independent
proxy clusters and are exploratory.
The secondary rho interval is [-0.1092,
0.3916], from 10000 defined cluster draws.
This uncertainty calculation was added during reporting for the frozen secondary
endpoint; it does not change membership, scoring, tests or the primary decision.

Fifteen triplets have complete stable boundaries, enough for the protocol's
tokenization-isolation *count* gate. Their post-inspection descriptive contrast is
-0.012716, interval [-0.021900, -0.006247].
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
