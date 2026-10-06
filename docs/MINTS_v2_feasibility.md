# MINTS v2 feasibility result

The frozen natural-variant protocol retained 0 matched cases
from 11 singleton heterozygous loci. Its
decision is **stop**. Native distribution scoring, real-case rescue
controls and head selection were not run because no matched case survived.
This measures control eligibility, not model sensitivity or biological absence.

## Population accounting

All 16 published rows were audited before model inference.

| Exclusion | Rows |
| --- | ---: |
| adjacent variants with unresolved phase | 4 |
| no sequence-only substitution/geometry matched sham | 4 |
| reference/alternate BPE boundaries differ | 7 |
| unidentifiable allele contrast | 1 |

The 7 boundary mismatches concern complete reference/alternate BPE offsets.
In 6 of these cases, token counts agree even though nucleotide boundaries
move, so matching tensor shape alone would not establish intervention alignment.
The 4 remaining singleton loci had no control meeting the frozen substitution,
context, geometry and motif-preservation rules. Neither matching tolerance nor
query policy was relaxed after this result. The engineered hook fixture passed,
but it does not demonstrate sensitivity of DNABERT-2 on natural variants.

## Why substitution matching failed

The deterministic search checked 512 candidate positions. Each row below
counts the first violated constraint in the frozen check order, not independent
failure rates or evidence that relaxing one rule would retain a valid control.

| First candidate rejection | Positions |
| --- | ---: |
| edit-token width mismatch | 5 |
| inside reference motif | 3 |
| local GC mismatch | 1 |
| query distance or side mismatch | 3 |
| trinucleotide mismatch | 500 |

These diagnostics use sequences and tokenizer offsets only. They add no model
outcomes, alternative endpoint, relaxed tolerance or confirmation cohort.

## Biological and confirmation limits

ADASTRA's metadata reports 80,735 CTCF candidate records. Its release is
accessible, but eligible independent donors/loci, source overlap and intervention
power remain unverified. Aggregate source counts do not authorize confirmation.
GSE81945 supplies pooled reads; phase, mapping-bias, dosage and replicate checks
remain missing. No native model prediction of binding direction is claimed.

There is no selected head or discovery intervention variance, so a powered
confirmation size cannot be estimated from this run. `power.json` labels its
normal-approximation scenarios illustrative. `confirmation_readiness.json`
keeps confirmation disabled; the old audit's inspected cohorts are excluded
from freshness claims.

## Reproduction and next boundary

This report was built from saved evidence with:

```powershell
python tools/build_variant_artifacts.py --study 'results/variant_study' --pilot 'results/variant_pilot' --report 'docs/MINTS_v2_feasibility.md'
python tools/audit_variant_study.py --study 'results/variant_study' --pilot 'results/variant_pilot'
```

Use fresh output directories for every scientific attempt, following the
[reproduction instructions](variant_verification.md#fresh-output-reproduction).
The report builder verifies the supplied study, scientific receipts,
historical attempt, metadata hashes and frozen Git baseline. The initial attempt
is retained with source snapshots; its reporting label was corrected to
distinguish unrun controls from failed controls without changing eligibility.

Continuing requires a new exploratory protocol for nucleotide correspondence
and control feasibility, or a larger cohort supporting the frozen rules.
Any revision must be saved before model scoring and must not reinterpret this
empty retained population as a positive result. The [protocol](MINTS_v2_protocol.md)
and [data inventory](variant_data_feasibility.md) describe the required evidence.
