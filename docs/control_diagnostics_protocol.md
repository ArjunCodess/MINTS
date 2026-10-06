# Sequence-control diagnostic v1

This exploratory method-development analysis follows inspection of the complete
ADASTRA Mabel v6.1 table. It cannot supply fresh confirmation. Its machine protocol
was written before execution in `results/control_diagnostics/protocol.json`.
The stopped natural-variant and native masked-flank protocols remain intact.

Select the first 12,288 allele-specific coordinate hashes with seed 1731 from all
512,556 coverage-eligible CTCF records, retaining nonsignificant records. Label the
first 4,096 as the original screen and the remaining 8,192 as an expansion. Use
204-base forward hg38 windows with the SNV at zero-based index 102. Verify the
reference allele and canonical sequence before evaluating any sequence predicates.
Use the existing JASPAR MA0139.1 both-strand scan and 0.8 support threshold.

Count all candidate positions within 64 bases, then count exact forward-strand
substitution candidates and exact-trinucleotide candidates. Evaluate the remaining
predicates independently on that explicit trinucleotide risk set, even when the
variant lacks a reference motif or complete allele token alignment. Record edit
width, query-side and distance constraints, local GC, motif-hit locations, motif
score preservation, sham tokenization and query correspondence as separate booleans.
Store every evaluated intersection at candidate level. These conditional predicate
rates are not independent failure probabilities on all source records.

Compare five declared sequence-only designs without any model outcomes: original
rules; identical query-span/ID correspondence in place of complete BPE alignment;
the latter without edit-token-width matching; and distance tolerances of 32 and 64
bases without width matching. The original decisions remain authoritative. The
diagnostic records are separate counterfactual eligibility assessments. Inspecting
these alternatives can justify a new exploratory protocol but cannot select an
endpoint by native score or establish native sensitivity.

Same-side queries lie outside the interval joining two single-base edits. The
absolute difference between their nucleotide distances to the query is exactly
the separation between edits. A 16-base tolerance therefore requires the sham
within 16 bases of the variant, despite searching a 64-base radius. This follows
from interval geometry; for single-base edits it is restrictive rather than
universally impossible. Exact trinucleotide, GC, motif exclusion and BPE-width
matching further intersect that small candidate set.

Reproduce with `python tools/run_control_diagnostics.py --output PATH` using a
fresh directory. The pinned tokenizer, cached hg38 and committed candidate table
are required; no model weights or biological effect columns are read. Budget
approximately 2 GiB spare disk, less than 20 MB outputs and minutes on a workstation.
