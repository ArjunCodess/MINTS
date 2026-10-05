# Native endpoint decision

The fixed masked-flank endpoint failed its discovery feasibility gate. Keep the
merged audit as the baseline, publish the pilot as a feasibility result, and stop
before searching heads or inspecting fresh confirmation data. Changing the
target, masking distance, direction, eligibility, or gate after this result starts
a new exploratory study.

## Literature comparison

[Ali, arXiv:2607.19618v1](https://arxiv.org/html/2607.19618v1), submitted July 21,
2026, trains sparse dictionaries and tests feature ablations using average
prediction-distribution KL shifts. It compares bound with GC-matched non-peak
motif windows, with random-feature and label controls. Sections 3.3 and 4 do not
specify a reproducible masking schedule, checkpoint commits, exact peak-file
accessions, or an independently held-out causal selection population. No code
repository link appears in the inspected HTML. These are reproducibility gaps
in that text, not proof that supporting material does not exist. A larger output
shift in bound windows establishes computational use associated with measured
binding, but does not itself show that the model predicts an allele's binding
change. MINTS therefore fixes one unchanged token target and keeps independent
allele-specific evidence as a separate biological test.

## Pilot protocol and result

`results/native_endpoint/protocol.json` was written before evaluation. It fixes
the checkpoint, MA0139.1 threshold, target rule, edit search, scan cap, clustering,
controls, and sensitivity gate. This is a local execution record, not an external
preregistration. Selection uses sequence geometry alone and never tries another
flank because the first eligible target has an unfavorable model score.

The target is the nearest unchanged whole downstream token with a 1 to 30 bp
gap from the motif, using upstream fallback only if no downstream target meets
the sequence-only eligibility rules. The clean, motif edit, and sham must have
identical offsets throughout the sequence, matching target IDs and nucleotides,
and an observable motif-token difference after masking. Both edits preserve
exact nucleotide counts and match base transitions, window width and token
budget. The primary contrast is clean-minus-motif log probability minus
clean-minus-sham log probability, in nats. A positive value means the motif edit
harms recovery more than the matched sham.

The pretrained prediction head loaded with no missing or mismatched weights.
The deterministic discovery sample retained 12 sequences on chromosomes 18/19,
forming 11 transitive 1 Mb genomic clusters. Its mean contrast was -0.0568 nats,
with a 95% cluster-bootstrap interval of [-0.2513, 0.0869]. The gate requires at
least eight clusters, a mean of at least 0.05 nats and an interval lower bound
above zero. It failed. No head was selected, including the prior trained-readout
candidate at layer 5, head 8. No confirmation was run. The wide interval supports
neither motif-specific recovery nor a precise absence claim.

## Conditions for a future frozen confirmation

Before opening new confirmation outcomes, save an immutable protocol with
membership hashes for a genuinely uninspected cohort, the chosen native endpoint,
one discovery selection rule, one selected component, the intervention query
positions, and one primary motif-minus-sham absolute rescue contrast. The planned
selection rule is the largest discovery mean contrast, with ties resolved by
layer then head; the planned intervention replaces the edited run's head context
at the masked query with the clean context. This is a proposed design, not a
completed or frozen experiment, because the endpoint has failed feasibility and
no native-output candidate exists.

Freeze exact offsets, target visibility, motif destruction, new-hit rejection,
matching tolerances, exclusions, duplicate/reverse-complement checks and genomic
clustering alongside the component. Record identity patches, whole-context clean
rescue, matched-sham rescue and the reverse intervention as sensitivity or
secondary checks. Verify identity invariance and whole-context recovery before
interpreting a head-specific null. Correct any secondary family as specified in
the frozen protocol, and do not select a secondary result as the new primary.

Chromosomes 20/21 were already inspected in the audit and cannot be reused as
fresh confirmation. Existing discovery data cannot establish freshness for a
new biological cohort. New inputs require an inspection ledger and genomic,
sequence-identity and homology separation checks before analysis.

Sample size must concern independent genomic clusters. For an initial planning
calculation, 80% power for a two-sided 5% test at a minimum contrast of 0.05 nats
requires approximately `ceil(((1.96 + 0.84) * sigma / 0.05)**2)` clusters, where
`sigma` is the discovery standard deviation of cluster-mean **intervention**
contrasts. This pilot's unpatched SD of 0.3128 would imply 307 clusters under the
equal-cluster normal approximation, but it is not an intervention variance or
a confirmation sample-size justification. A passing discovery intervention must
provide that variance, retention rates and cluster-size distribution, followed
by power simulations using the frozen estimator. Do not use 64 sequences as a
substitute for 64 independent clusters.

A confirmed rescue supports a bounded native sequence-prediction claim. A null
supports an absence claim only if demonstrated sensitivity and its interval
exclude the frozen meaningful effect. Neither result alone certifies CTCF binding.

## Independent biological anchor

[GSE81945](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81945) reports
allele-specific reduction of CTCF binding at mutant motifs in melanoma. The
[public SOFT metadata](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE81nnn/GSE81945/soft/GSE81945_family.soft.gz)
lists GSM2178295 and GSM2178296 as CTCF ChIP replicates in COLO829 and GSM2178297
as IgG. Alignments use hg19; the processed replicate files contain peak calls.
The checked metadata does not supply a paired variant/binding-effect table, so
the usable allele-specific sample size remains unverified. Peak count is not
variant sample size, and hg19 coordinates cannot be read directly as hg38.

A biological extension needs reference/alternate alleles, genome-build-verified
windows, replicate allele-specific counts, input-normalized binding effects and
mapping-bias checks. Obtain the study's variant table and linked sequencing
inputs before deciding eligibility or power. Match natural substitutions by
reference-to-alternate change, local context, motif position and token geometry;
exact-composition shuffles cannot replace those controls. Report agreement with
binding effects separately from native-token recovery, including disagreement.
