PR title: Resolve XAI4Science review issues and harden MINTS evidence

Summary
- audit and correct numerical, sample-count, and result-table inconsistencies identified during XAI4Science review
- strengthen leakage controls, sequence-level uncertainty, multiple-comparison handling, and motif-absent controls; report the new native-attention associations separately from historical screen failure
- rerun TATA patching on held-out examples, narrow causal claims to the trained probe, and remove the unvalidated Nucleotide Transformer comparison
- rewrite the manuscript for clearer motivation, methods, claim boundaries, and reproducibility

Validation
- 53 tests pass; the revised seven-page PDF compiles, renders, and passes source/text/metadata/annotation screening
- instrumented commands, exit statuses, runtimes, failures, and logs: [command history](COMMAND_HISTORY.md); earlier discovery-command recording limitations: [validation record](VALIDATION.md)
- every regenerated numeric table and figure maps to saved machine-readable sources in [validation record](VALIDATION.md) and [artifact hashes](../results/review/final_artifact_manifest.json)
- clean installation and ingestion succeeded, but the full pipeline failed during ENCODE download with disk exhaustion; native CTCF causal patching, sequence-level QK inference, and upstream provenance remain unresolved, as recorded in [claim audit](CLAIM_TO_EVIDENCE_AUDIT.md)
