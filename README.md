# MINTS

MINTS asks what evidence is required before calling a genomic-transformer attention head a biological motif detector. Frozen label decodability, motif-local association, and effects on a trained probe support different claims.

The exact submission is preserved by tag `xai4science-submission-2026`, commit `7112ec22b770c1651f24d33ac5f45fda3e986324`, confirmed by the author. Review work is on `xai4science-review-improvements`. Revised outputs are in `results/review/`; historical results remain preserved.

See [response matrix](docs/XAI4SCIENCE_REVIEW_RESPONSE_MATRIX.md), [claim audit](docs/CLAIM_TO_EVIDENCE_AUDIT.md), [source verification](docs/SOURCE_VERIFICATION.md), and [manuscript source](paper/main.tex).

## Reproduce the review

```powershell
python tools/review_audit.py
python tools/review_tables.py
python tools/review_patching.py
python tools/review_ctcf_controls.py
python tools/review_manuscript_artifacts.py
python -m pytest -q -p no:cacheprovider --basetemp results/review/test-tmp-new
python tools/review_submission.py
```

These commands require dependencies in `requirements.txt`. The audit and tables read the existing saved datasets, token-score CSVs, and frozen activation caches. They are not a clean encoder reproduction. Patching reads the training cache and saves test-only effects separately. Artifact generation requires that patching has completed. Actual versions, seeds, runtime, hashes, and failures accompany the outputs.

The original uncapped pipeline is `python main.py`. A clean full attempt using an isolated data/result tree is `python tools/review_full_pipeline.py`. Preserve a failed run before retrying in a new directory. Record commands with `python tools/review_command.py python tools/review_full_pipeline.py`.

## Findings and claim boundaries

The former Seq-head and Readout columns were the same classifier. Historical unconstrained matching retained both complete promoter test sets; the revised GC caliper is a post-review heuristic. Motif counts distinguish unique tokens from occurrences in overlapping motif hits. Historical patching pooled train and test examples; revised patching uses test only and saves pair effects and denominators. Its target is the trained probe decision function, not native behavior. Head maxima remain exploratory.

No head passed the historical CTCF QK and reconstructed-attention screens. A separate pinned native-attention experiment validates reconstructed head contexts and finds conditional motif-present/absent associations across matched sequence pairs, with Holm correction across 144 heads. Sequence-level QK inference and a native CTCF intervention target remain unresolved. The Nucleotide Transformer comparison is excluded from the manuscript. The clean full pipeline failed during an ENCODE download when the disk filled; it has not been completed.

## Inputs and terms

Downstream tasks use `InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`. Saved partitions contain train/test, with test chromosomes 20/21 and no independent validation set. Membership JSONL records contain coordinates, labels, and hashes. CTCF uses ENCSR000DKV / ENCFF827JRI, UCSC hg38, and JASPAR MA0139.1.

Original MINTS code is MIT licensed. Third-party data, models, and the venue style file retain their own terms. Checkpoint and upstream dataset revisions were unpinned in the submission; current reruns must record their actual snapshots without claiming historical identities.
