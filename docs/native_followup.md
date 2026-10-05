# Native endpoint follow-up

These diagnostics investigate the stopped discovery pilot without changing its
endpoint, selecting heads, or opening a confirmation analysis. Scientific
source hashes, fixed memberships and execution outputs are preserved in
`results/native_followup/`; the original `results/native_endpoint/` is unchanged.

## Geometry and pilot influence

Shams must be disjoint from the motif, have the same width, lie on the same side
of the fixed masked target, and match its nucleotide gap within two bases.
Two disjoint 19-base windows on the same side have gaps differing by at least
19 bases. All 12 original cases are therefore structurally infeasible. The
runner records that exclusion before scoring and never relaxes the rule or
changes the target. A future comparison across sequences would need a new
protocol because it also changes target labels and surrounding context.

The eligibility audit covers all 21 scanned windows, including the nine
excluded windows, and reports length, GC, token geometry and continuous maximum
PWM score. Removing each whole genomic cluster changes the pilot mean from
-0.0922 to +0.0163 nats across omissions. This sign sensitivity and the original
wide interval prevent a precise absence claim.

## Recovery competence and implementation controls

Before execution, the protocol fixes 128 chromosome 16/17 windows, one random
real masked token per window, and seed 1730. A smoothed frequency baseline uses
512 fixed chromosome 14/15 windows. No model output filters membership. Native
mean log probability is -6.0632 versus -6.8769 for the frequency baseline;
median vocabulary rank is 57.5 and top-one recovery is 6.25%. Width and frequency
strata accompany these aggregate scores because recovery varies strongly with
token geometry. This demonstrates some context-dependent recovery competence,
without establishing sensitivity to CTCF edits.

All 24 motif/sham cases pass native hook identity, complete clean final-residual
rescue and reverse edited-residual checks within a 0.0002 maximum-logit error.
Attention-context-only rescue is recorded separately because residual paths
can bypass those contexts. An engineered motif-dependent predictor changes
its native target score by 0.6750 nats and recovers it exactly through the same
hook path, providing a positive test of the intervention machinery.

## Measured allele data

The official Poulos et al. supplement contains measured allele counts in
**Table S3 of mmc1.pdf**. The article's Table S2 reference is inconsistent with
the supplement, whose Table S2 lists RAD21 accessions. The workbook mmc2.xlsx
contains genomic site lists rather than the measured variant effects.

The extracted table contains 16 variants, including 15 heterozygous variants
at 13 loci and one homozygous locus that cannot identify an allele contrast.
Thirteen heterozygous variants at 11 loci lie in the stated strict motif core.
All 16 one-based source reference alleles match UCSC hg19; saved 204-base
reference/alternate windows use zero-based half-open coordinates. Two loci
contain adjacent mutations whose phase is unresolved, so their variants must
not be treated as independent observations.

Normalized mutant ChIP fractions are reconstructed from WGS-normalized counts
and checked against the source's rounded values. Half-count-corrected log odds
intervals describe pooled read-count variation; they exclude replicate
variation, mapping bias and cell-line uncertainty. No hg38 liftover,
substitution-matched population, native model binding prediction or biological
confirmation is claimed. Publisher sources retain their third-party terms.

## Reproduction

Use a fresh output directory for another extraction and scientific run:

```powershell
python tools/extract_ctcf_alleles.py --output results/native_followup_new
python tools/run_native_followup.py --output results/native_followup_new --device cuda
```

The scientific runner refuses an existing protocol to preserve completed
evidence. `python tools/build_native_followup.py` verifies the saved default
run and regenerates its manuscript insert, figures and eligibility summaries.
The report manifest hashes those derived outputs separately from the immutable
scientific execution. Tests include structural geometry exclusion, deterministic
sampling, hook cleanup, engineered rescue, allele normalization, locus grouping
and source/artifact consistency.
