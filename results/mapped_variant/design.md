# Mapped-query exploratory protocol v1

This is a new exploratory single-SNV reference-scenario study chosen from the
sequence-only diagnostic, before native scoring. ADASTRA, GSE81945 and all
earlier audit cohorts remain inspected. No individual haplotypes are inferred.
The prior stopped protocols and their decisions remain unchanged.

## Scientific question and estimand

On the retained motif-overlapping reference scenarios, does the natural variant
change local native MLM distributions more than a paired motif-preserving
substitution? The primary endpoint is variant-minus-sham Jensen-Shannon divergence
in nats, averaged within case by query nucleotide width, then equally over genomic
cluster means. This measures sensitivity to the declared sequence perturbations.
It does not identify binding direction, binding affinity or a CTCF circuit.
Different segmentation outside queries remains part of the measured perturbation.

## Population, selection and controls

Use exactly the 12,288 members of the preceding diagnostic, retaining every source
significance level. Freeze membership hashes before model loading. Source denominator
is 512,556. Exclude invalid/out-of-range reference windows, noncanonical DNA,
reference mismatches, absent reference CTCF motif, absent shared allele queries,
then absence of a matched sham. Require forward hg38 204-base windows, index 102,
exactly one reference-to-alternate SNV, and MA0139.1 at the existing 0.8 support
threshold. Choose the highest-scoring reference hit covering the SNV, breaking
ties by position; scan both strands. The threshold retains continuity in the
motif definition rather than optimizing model effects.

Search all within-window shams within 16 bases in ascending distance then position.
Require identical forward substitution and trinucleotide, outside every reference
motif hit, local 33-base GC difference at most 0.05, unchanged motif-hit locations,
and selected motif score change at most 0.1 bits. Keep these existing composition,
position and motif controls. Complete input BPE alignment and edit-token width
are removed because the diagnostic retains only 1 and 11 cases respectively,
versus 61 with both removed. Query span and token identity remain exact. This is
a paired within-window control, with no shared shams between cases. Matching
does not establish biological exchangeability or isolate motif effects from
different edit-segmentation responses.

## Query task and interventions

Select every real token within 32 bases of the variant, outside every reference
motif hit and outside the interval joining the edits, whose half-open nucleotide
span and vocabulary identity agree in reference, alternate and sham. Map its
three indices explicitly. Mask each separately after tokenization, without
retokenizing. The target token ID and nucleotide string are identical, so all
three distributions predict the same vocabulary-valued query task. Both edits
remain visible. The other tokens, token counts, query indices and ALiBi distances
can differ; these are logged and constrain interpretation, even with equal
nucleotide geometry.

A head intervention, if authorized by the gates below, transfers only the reference
pre-projection head-context channels at that mapped query into the recipient's
mapped query at the declared layer. No equal-index or whole-tensor transfer is
permitted. A final-query representation restoration control copies the final
encoder row into the recipient mapped row. The native MLM head is pointwise, so
that control is mathematically expected to recover reference query logits exactly
up to numerical tolerance. An attention-head transfer has no such expectation.

## Implementation, statistics and gates

Use pinned DNABERT-2 revision `7bce263b15377fc15361f52cfab88f8b586abda0`, identical
tokenizer revision, evaluation mode, float32 on CUDA with TF32 disabled, deterministic
PyTorch operations, and float64 JS arithmetic. Verify native loading without
missing prediction weights. Save software, checkpoint/cache file hashes, source
hashes, query maps, scores and failures. Maximum logit error of identity, repeat,
mapped final restoration and reverse restoration must be at most 0.0002. This
existing wiring tolerance concerns unchanged computations rather than a new
scientific effect threshold. Normalization error must be at most 1e-12.

Feasibility requires at least eight transitive genomic/sequence clusters, with all
retained cases implementation-valid. There is no source retention target because
the estimand explicitly concerns the small eligible population. Failure stops
scoring/head progression without changing constraints. Group cases whose 1 Mb
expanded genomic blocks overlap transitively or whose reference/alternate/sham
sequences are identical or reverse complements. These are dependence proxies,
not independently verified donors. Save cluster membership. Bootstrap equal-weight
cluster means 10,000 times, seed 1731, to obtain the primary 95% percentile interval.
Report leave-one-cluster-out estimates and chromosome-cluster sensitivity.

Native sensitivity advancement requires a positive primary lower interval bound;
no biological minimum worth detecting has been established, so no old 0.001-nat
threshold is imported. A positive result is a bounded exploratory association.
Secondary tests form one Holm family: association of variant divergence with
absolute PWM change, and a contrast on cases with matched reference edit-token
width. Report effect sizes and uncertainty on the same selected membership, without
using association to select cases. Also report full-boundary-stable counts,
query-index differences and token-count-change differences. These descriptive
diagnostics cannot remove residual tokenization confounding.

Causal head search requires native sensitivity, implementation validity, at least
eight clusters with complete boundary-stable triplets, and biological QC with
traceable experiment/biosample/donor, genotype/phase, replicate ChIP/input counts,
mapping correction and dosage handling. These conditions are necessary here to
distinguish generic tokenization response from the intended motif mechanism.
If any fail, stop with the completed exploratory sensitivity result. Do not search
heads or compute confirmation power from unpatched variance. A future design may
address these limits but must be separately frozen.

## Biological comparison and confirmation

Investigate publisher schema and experiment metadata without fabricating counts.
ADASTRA weighted allele-wise log2 observed/expected effects are not pooled
input-normalized log odds and lack the per-case provenance needed by this inference.
No biological correlation is authorized until the QC gate passes. Direct motif
overlap here is a sequence definition, not evidence of direct in vivo binding.
Donor counts, QC-eligible biological cases and independent donors remain unknown
unless verified. Keep missing quantities null rather than inventing zeros.

Independent confirmation is unassigned and cannot use the inspected source.
It requires a frozen discovered intervention, its cluster variance, a justified
minimum effect and power, plus donor/experiment/publication, coordinate, exact/RC
sequence and relevant homology separation. Failure to meet the head gates ends
this study without a confirmation protocol or mechanistic null claim.
