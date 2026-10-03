# Validation and reproducibility status

The revision is tested and locally compiled, but the clean full scientific pipeline did **not** complete. `results/review/commands.jsonl` is the exact argv/exit-status/runtime ledger for instrumented executions, including failures and retries. Associated `command_*.log` files preserve output. `tools/review_command.py` reproduces this recording behavior.

## Executed scientific commands

- `.venv/Scripts/python.exe tools/review_audit.py`: exit 0; local membership, leakage, near-duplicate, count, GC, patching and metric audit.
- `.venv/Scripts/python.exe tools/review_tables.py`: exit 0; classifier refits and 1,000 sequence-bootstrap intervals, seed 1729, 130.313 seconds.
- `.venv/Scripts/python.exe tools/review_patching.py`: exit 0 on reruns; pinned CUDA test-only TATA patching. Latest raw-effect rerun took 27.748 seconds.
- `.venv/Scripts/python.exe tools/review_ctcf_controls.py`: exit 0; 5,220 matched sequence pairs, 144 heads, 9,999 paired sign flips and 1,000 pair bootstraps, seed 1729. Internal runtime 241.073 seconds. Native context maximum error 0.
- `.venv/Scripts/python.exe tools/review_manuscript_artifacts.py`: final execution exit 0. Earlier syntax and Series-attribute errors are logged and corrected.
- `.venv-review/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp results/review/test-tmp-clean-2`: exit 0, 53 tests passed. Clean dependency versions are saved before the temporary environment was removed to recover disk space.
- `tools/review_full_pipeline.py` in the isolated clean environment: exit 1. Dependency installation initially failed, explicit PyPI installation succeeded, and downstream ingestion completed. ENCODE download then failed with `OSError: [Errno 28] No space left on device`. The failure manifest and log are preserved. No full-pipeline success is claimed.

Final test, scanner, compile, PDF inspection, and diff-check executions appear in the command ledger. `requirements-review-existing.lock`, `requirements-review-clean.lock`, and the two environment-version JSON files record actual package versions, rather than asserting identical environments. The historical model revision is unpinned; new patching/control reruns pin revision `7bce263b15377fc15361f52cfab88f8b586abda0`.

## Manuscript artifact mapping

| Output | Machine-readable source | Reproducible generator |
|---|---|---|
| Table 1 evidence criteria | Editorial definitions; no experiment numbers | `paper/main.tex` |
| Table 2 partition counts | `correctness_audit.json`, four membership JSONL files | `tools/review_audit.py`; `tools/review_manuscript_artifacts.py` |
| Tables 3 and 4 AUROC/intervals | `classification_metrics.csv`, four prediction CSVs, GC diagnostics | `tools/review_tables.py`; `tools/review_manuscript_artifacts.py` |
| Table 5 historical screen sweep | Submitted QK/enrichment CSVs; `ctcf_threshold_sweep.csv` | `tools/review_manuscript_artifacts.py` |
| Table 6 test-only patching | Held-out head CSV and pair-effect NPZ | `tools/review_patching.py`; `tools/review_manuscript_artifacts.py` |
| Table 7 resource terms | Official source checks in `SOURCE_VERIFICATION.md` | Editorial table; unresolved licenses explicitly marked |
| Native-attention figure | `ctcf_native_control_inference.csv`, raw score NPZ, matched pairs | `tools/review_ctcf_controls.py`; `tools/review_manuscript_artifacts.py` |
| Patching pair figure | `heldout_pair_plot_data.csv`, raw pair-effect NPZ | `tools/review_manuscript_artifacts.py` |
| Numeric macros | Correctness audit, saved historical head tables, native inference CSV | `tools/review_manuscript_artifacts.py` |

Final output hashes and source mapping are in `final_artifact_manifest.json`. Historical submitted outputs remain inspectable at the preserved tag; `paper/main.pdf` is replaced only after compiling the revised source.

## Recording limitations and failed attempts

Exploratory shell reads and initial Git/skill discovery preceded the command wrapper. Their complete tool transcript is the authoritative record; this ledger must not be described as a retrospective record of every shell command. Initial `git status --short` was clean. Unprivileged Git tag/switch attempts failed with access denial, then authorized escalated tag creation and branch switch succeeded. Read-only searches included `rg`, `Get-Content`, `Get-ChildItem`, Git log/show/status and environment/model-source inspection. No original changes were discarded.

The first logging wrapper used a serial filename, and two concurrent commands collided in `command_000.log`. Later scientific reruns have unique UUID logs. Initial dependency installation, a missing-numpy full-run attempt, artifact-generator errors, and the disk-full pipeline failure remain recorded. Temporary cache/environment cleanup initially encountered file-access failures, then succeeded with verified workspace-contained paths; original `.venv`, datasets, caches and historical results were preserved. The clean environment was removed only after saving versions and successful test output.

Before committing on 2026-10-03, the scientific review artifacts and manuscript were rebuilt, and all 53 tests passed again. A first membership rewrite failed with a transient Windows `Invalid argument` error; the audit retry completed successfully. Model-loading logs include a blocked optional Hugging Face safetensors-conversion request; the pinned local checkpoint loaded and patching completed with exit 0. These successful review builds do not turn the earlier failed full clean pipeline into a successful reproduction.

Submission scans cover the actual manuscript sources/inserts and final PDF, including metadata and annotations. Internal review documents and preserved historical artifacts intentionally contain the review history and author identity, so they are not part of an anonymous submission package. Final scientific interpretation still requires the author's review.
