# MINTS

MINTS asks what evidence is required before calling a genomic-transformer attention head a biological motif detector. Frozen label decodability, motif-local association, and effects on a trained probe support different claims.

The exact submission is preserved by tag `xai4science-submission-2026`, commit `7112ec22b770c1651f24d33ac5f45fda3e986324`, confirmed by the author. Review work is on `xai4science-review-improvements`. Revised outputs are in `results/review/`; historical results remain preserved.

See the [paper](paper/main.pdf) for methods, results, limitations, references, and resource licenses.

## Build the paper

The paper inputs [results.tex](paper/results.tex), which contains its numerical variables and table bodies. Generate it and the figures from saved results, then compile normally:

```powershell
python tools/review_manuscript_artifacts.py
latexmk -pdf -cd -interaction=nonstopmode -halt-on-error -outdir=review-build paper/main.tex
Copy-Item -LiteralPath paper/review-build/main.pdf -Destination paper/main.pdf
```

The generator requires dependencies in `requirements.txt` and the saved CSV/JSON/NPZ results. Its output hash and source command are recorded in [the table manifest](results/review/tables_manifest.json).

## Rerun the analyses

```powershell
python tools/review_audit.py
python tools/review_tables.py
python tools/review_patching.py
python tools/review_ctcf_controls.py
python -m pytest -q
```

These commands require dependencies in `requirements.txt`, saved datasets, token scores, and activation caches. The audit and classifier refits reuse saved inputs; patching and genomic controls run the model. Actual packages used are recorded in the [CUDA environment](results/review/existing_environment_versions.json) and [clean CPU environment](results/review/clean_environment_versions.json).

`python main.py` runs the full pipeline. `python tools/review_full_pipeline.py` uses an isolated output tree and refuses to overwrite a previous run. The earlier clean run failed during ENCODE download with disk exhaustion; its [failure manifest](results/review/failed_full_pipeline_manifest.json) and [historical command ledger](results/review/commands.jsonl) remain available. The ledger covers instrumented runs, not every exploratory command.

Inputs come from the upstream [DNABERT-2 checkpoint](https://huggingface.co/zhihan1996/DNABERT-2-117M), [revised downstream dataset](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised), and [JASPAR matrix](https://jaspar.elixir.no/matrix/MA0139.1/). ENCODE download URLs and the reference-genome URL are recorded in `data/ENCODE4_v1.5.1_GRCh38.txt` and `src/config.py`.

Original MINTS code is [MIT licensed](LICENSE); third-party resources retain their own terms.
