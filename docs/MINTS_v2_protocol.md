# MINTS v2 natural-variant feasibility protocol

**Status.** Exploratory, locally frozen before model scoring, not external
preregistration. The merged v1 audit remains immutable. Failure closes this
protocol without head search, endpoint substitution or tolerance relaxation.

**Hypothesis and endpoint.** CTCF-overlapping substitutions produce greater
local native prediction-distribution change than sequence-matched,
motif-preserving substitutions. Use the pinned pretrained DNABERT-2 MLM with
no fitted readout. The new endpoint is the nucleotide-width-weighted mean
Jensen-Shannon divergence between reference and alternate vocabulary
distributions over unchanged local tokens. Mask each query separately,
retaining the variant in the input. Include every eligible query within 32 bp,
outside reference motif hits and the interval joining variant and sham edits.
Require identical complete token offsets and unchanged query IDs across all
three sequences. This symmetric endpoint measures sensitivity, not which
allele binds more strongly, and does not revive v1's failed single-target score.

**Data and eligibility.** Pilot all 16 published GSE81945 Table S3 rows in hg19.
Require canonical 204-base windows, verified reference alleles, exact
single-substitution orientation, heterozygosity, finite integer counts, and a
reference MA0139.1 hit at the fixed 0.8 support threshold containing the variant.
Exclude the homozygous row and all rows at the two unresolved adjacent-variant
loci, leaving at most 11 singleton loci before token/control exclusions.
Alternate alleles are supported by the publication, not by the reference genome.
No liftover is performed. Pooled counts remain exploratory; confirmation needs
phase, replicate, mapping-bias, dosage and donor QC.

**Controls.** Search reference-window positions deterministically by distance,
then coordinate, within 64 bp. Require the same forward-strand substitution,
exact trinucleotide, edit-token width, local 33-base GC within 0.05, matching
query-side geometry with distance differences at most 16 bp, unchanged motif-hit
locations, and motif score changes at most 0.1 bits. Shams lie outside all
reference motif hits. Choose the first eligible sham before model inference.
Do not match away the variant's motif disruption. Record every exclusion.

**Pilot statistics and gates.** Resample equal-weight 1 Mb genomic-cluster
means with 2,000 bootstrap draws, seed 1731. Require at least eight clusters,
retention of at least 50% of singleton heterozygous loci, implementation controls
within 0.0002 maximum logit error, mean variant-minus-sham divergence above
0.001 nats and a positive 95% lower bound. Also require Spearman correlation
above 0.3 between variant divergence and absolute input-normalized binding
log odds, with a positive cluster-bootstrap lower bound and at least 95%
defined bootstrap draws. Compare descriptive PWM-change magnitude agreement
on identical membership. These numerical gates are engineering choices, not
established biological effect thresholds. Stop on any failure.

**Discovery and confirmation.** Head search is conditional on every pilot gate
passing. Select the largest discovery mean variant-minus-sham absolute rescue,
breaking ties by layer then head. Patch only that head's pre-projection context
at each masked query from reference into alternate/sham. Freeze the head and
intervention after discovery; candidate maxima provide no confirmatory p-value.
Confirmation membership is currently unassigned. All existing audit cohorts and
GSE81945 are inspected; ADASTRA has only aggregate metadata inspection so far,
but source overlap and donor dependence remain unresolved. Require coordinate,
exact/RC, homology, donor and source separation before opening new outcomes.

**Power and final inference.** Estimate noise from discovery *intervention*
cluster means and simulate 80% power for a two-sided 5% test at 0.001 nats,
accounting for retention. No unpatched variance substitutes for this estimate.
Biological agreement requires its own power/design justification. A future
immutable confirmation protocol must name the new cohort, membership hashes,
chosen head, numerical thresholds, biological QC and justified sample size.
Use one primary head-rescue-minus-sham effect and cluster-level inference;
apply Holm correction to a declared secondary family. If those requirements
cannot be met, retain feasibility findings and end the study.
