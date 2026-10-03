# Build and validation

Run the artifact-generation commands in [README](../README.md) from the repository root, using dependencies from `requirements.txt`. The two `requirements-review-*.lock` files capture the tested environments; the existing snapshot used CUDA, while the clean snapshot used CPU. The review generators require the saved datasets, token scores, and activation caches.

## Build the paper

Use an existing LaTeX installation, then copy the compiled PDF to its published repository path:

```powershell
latexmk -pdf -g -cd -interaction=nonstopmode -halt-on-error -outdir=review-build paper/main.tex
Copy-Item -LiteralPath paper/review-build/main.pdf -Destination paper/main.pdf
python tools/review_pdf.py
python tools/review_submission.py
python tools/review_finalize.py
```

PDF inspection requires a Python environment with `pypdf`; the recorded run used the bundled document runtime. The scanner verifies PDF text, metadata, annotations, and generated tables. Finalization hashes outputs and builds the anonymous source archive. [The artifact manifest](../results/review/final_artifact_manifest.json) maps tables and figures to their saved sources.

## Full pipeline and execution records

`python main.py` runs the original pipeline. `python tools/review_full_pipeline.py` runs it in an isolated data/result tree and refuses to overwrite an existing run. Preserve a failed run and use a new directory before retrying.

Record any execution with `python tools/review_command.py <command> <arguments>`. [The command ledger](../results/review/commands.jsonl) stores argv, exit status, runtime, and log paths. Initial exploratory commands preceded logging; the first serial log also suffered a filename collision, so subsequent runs use unique filenames.

All 53 tests passed and the revised PDF compiled successfully. The clean full pipeline failed during ENCODE download with disk exhaustion after installation and ingestion succeeded; [its failure manifest](../results/review/failed_full_pipeline_manifest.json) and logs are retained. The temporary clean environment was removed after saving package versions and test output. These review builds are not a completed clean encoder reproduction.
