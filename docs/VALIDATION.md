# Build

The paper loads `paper/results.tex` once. It defines the numerical variables and table bodies; figures use the saved PNGs in `results/review/`.

To regenerate these from the saved CSV/JSON/NPZ results and build the PDF:

```powershell
python tools/review_manuscript_artifacts.py
latexmk -pdf -cd -interaction=nonstopmode -halt-on-error -outdir=review-build paper/main.tex
Copy-Item -LiteralPath paper/review-build/main.pdf -Destination paper/main.pdf
```

No PDF-extraction or packaging step is required. [The table manifest](../results/review/tables_manifest.json) records the generated TeX file's source command and hash.

To rerun the scientific analyses, use the commands in [README](../README.md). They require the saved datasets, activation caches, token scores, and dependencies in `requirements.txt`. The two dependency snapshots capture the tested CUDA and clean CPU environments. Run tests with `python -m pytest -q`.

`python main.py` runs the full pipeline; `python tools/review_full_pipeline.py` uses an isolated output tree. The earlier clean run failed during ENCODE download with disk exhaustion after installation and ingestion succeeded. Its [failure manifest](../results/review/failed_full_pipeline_manifest.json), [command ledger](../results/review/commands.jsonl), and logs remain available. The ledger covers instrumented historical runs, not every exploratory command.
