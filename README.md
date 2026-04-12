# MINTS

**Mechanistic Interpretability of Nucleotide Transformer Sequences**

## Reproducible Pipeline

Install the project dependencies separately from the pipeline run:

```powershell
python -m pip install -r requirements.txt
```

Then run the complete pipeline from one command:

```powershell
python main.py
```

The run creates a manifest at `results/manifests/pipeline_run.json`.

For a full overwrite of generated outputs:

```powershell
python main.py --overwrite
```

The pipeline writes intermediate data under `data/` and manifests under `results/manifests/`.
