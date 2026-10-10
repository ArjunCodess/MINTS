# MINTS

MINTS audits nucleotide correspondence and control feasibility in genomic transformer interventions. Equal tensor shapes can hide mismatched nucleotide positions. Explicit span and vocabulary correspondence permits comparison of a specified prediction task across changing BPE segmentation, while surrounding segmentation and context can still differ.

The [main paper](paper/main.pdf) and [supplement](paper/supplement.pdf) report a repaired CTCF component effect on a fixed trained readout, inconclusive TATA results and failed native sensitivity gates. In the 61-case mapped-query study, sham prediction shifts exceed variant shifts. These results do not establish biological binding causality or general sensitivity to learned mechanisms.

GLBIO 2027 full-paper proceedings is the chosen first review. The [venue presentations](paper/venues/README.md) include its OUP manuscript and conditional journal/community candidates. [Submission routes](docs/submission_routes.md) record verified deadlines, required author information, fee dependencies and overlap checks. Preparation does not activate a submission.

## Reproduce the public evidence and paper

Use Python 3.14 and the pinned CPU dependencies. These commands audit saved evidence and rebuild numerical inserts and figures without model or genome downloads:

```powershell
python -m pip install -r requirements-cpu.txt
python -m pip check
python tools/audit_variant_study.py
python tools/audit_adastra_feasibility.py
python tools/audit_mapped_variant.py
python tools/check_query_correspondence.py
python tools/build_paper_artifacts.py
python -m pytest -q -m "not artifact" --basetemp .test-tmp
python -m pytest -q -m artifact --basetemp .test-tmp/artifact
```

Compile both papers with an installed LaTeX environment from `paper`:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=review-build main.tex supplement.tex
```

The [correspondence contract](docs/query_correspondence.md) explains its guarantees and limits. The dependency-light example uses hand-constructed maps; it is not model calibration. A separately versioned post-study certificate reproduces all 438 saved queries without changing scientific membership, predictions or gates. Retokenizing with `--saved-cases --output NEW_PATH.json` requires the pinned tokenizer and JASPAR cache, but no model weights.

Original protocols, source receipts, timestamps and stopped studies remain intact under `results`. Historical manuscript hashes are checked against the original Git revision, not against the updated paper. `paper/figures/lineage.json` records input, generator and output hashes for the public presentation rebuild. The supplement and its build inputs are public. The [versioned reproduction dataset](https://doi.org/10.5281/zenodo.23267982) is public under CC-BY-4.0 and includes all 61 unchanged numeric logit files. Download `mints-results-v1.zip`, extract it into `DATA_DIR`, then run `python tools/verify_deposit.py --data-root DATA_DIR` from this checkout. This verifies hashes, headline estimates and corrections using saved clusters. It does not independently reconstruct excluded sequence inputs or rerun model inference. The complete mixed-resource reviewer archive remains private.

The [venue strategy](docs/venue_strategy.md) explains the journal route and its unresolved significance bar.

## Scientific execution

`python main.py` is the historical pipeline; withdrawn patch rankings remain diagnostic evidence. Dedicated hardened/native/mapped runners implement the recorded assay definitions. Full execution requires the pinned checkpoint and tokenizer, verified native prediction weights, genome, intervals and residual caches:

```powershell
python tools/preflight.py --scientific
python tools/run_hardened_assay.py --help
python tools/run_native_endpoint.py --help
python tools/run_mapped_variant.py --help
```

New experiments require a separately frozen protocol and a specified claim. Report reproduction does not reopen stopped studies or require model reruns.

## Licenses

Original software is [MIT](LICENSE). Original scientific outputs are [CC-BY-4.0](LICENSE-OUTPUTS). [License scope](docs/licensing.md) separates these grants from upstream code, models, datasets and reference resources; third-party permissions are not granted by MINTS.
