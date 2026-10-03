# MINTS

MINTS asks what evidence is required before calling a genomic-transformer attention head a biological motif detector. Frozen label decodability, motif-local association, and effects on a trained probe support different claims.

The exact submission is preserved by tag `xai4science-submission-2026`, commit `7112ec22b770c1651f24d33ac5f45fda3e986324`, confirmed by the author. Review work is on `xai4science-review-improvements`. Revised outputs are in `results/review/`; historical results remain preserved.

See the [paper](paper/main.pdf) for methods, results, and limitations, [build instructions](docs/VALIDATION.md), and [source verification](docs/SOURCE_VERIFICATION.md).

## Reproduce the review

```powershell
python tools/review_audit.py
python tools/review_tables.py
python tools/review_patching.py
python tools/review_ctcf_controls.py
python tools/review_manuscript_artifacts.py
python -m pytest -q -p no:cacheprovider --basetemp results/review/test-tmp-new
```

These commands require dependencies in `requirements.txt`. The audit and tables read the existing saved datasets, token-score CSVs, and frozen activation caches. They are not a clean encoder reproduction. Patching reads the training cache and saves test-only effects separately. Artifact generation requires that patching has completed. Actual versions, seeds, runtime, hashes, and failures accompany the outputs.

The paper inputs [results.tex](paper/results.tex), which contains its numerical variables and table bodies. Regenerate that file and the figures from saved results with `python tools/review_manuscript_artifacts.py`, then compile normally. Original MINTS code is [MIT licensed](LICENSE); third-party resources retain their own terms.
