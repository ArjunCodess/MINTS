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

By default, this uses all available rows for residual caching and linear probing. The run creates the root summary manifest at `results/pipeline_run.json`.

For a full overwrite of generated outputs:

```powershell
python main.py --overwrite
```

For a faster capped run while debugging, limit the number of train and test rows used for activation caching and linear probing:

```powershell
python main.py --max-probe-train 512 --max-probe-test 256
```

The caps are applied per task. Omit those flags to run probing on the full saved train/test splits.

The pipeline writes intermediate data under `data/`, detailed component manifests under `results/manifests/`, and the commit-friendly run summary under `results/pipeline_run.json`.
