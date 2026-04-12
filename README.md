# MINTS

**Mechanistic Interpretability for Nucleotide Transformer Sequences**

**TL;DR:** MINTS is a reproducible mechanistic-interpretability pipeline for genomic transformers. It downloads nucleotide benchmark data, wraps DNABERT-2 with the best available hook backend, extracts QK/OV circuit matrices, probes frozen residual-stream features, generates motif-destroying counterfactuals, and provides activation-patching utilities for testing whether candidate heads causally restore biological signal.

MINTS is built around one question: can a nucleotide transformer be studied strongly enough that a motif detector is not just visualized, but supported by converging circuit, probing, enrichment, and causal-intervention evidence? The repository currently targets:

- `hf_downstream`: promoter and splice-site tasks from `InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`
- `ctcf_gm12878`: ENCODE GM12878 CTCF artifacts and derived GRCh38 peak sequences

The research paper lives in [`paper/main.pdf`](paper/main.pdf), with source in [`paper/main.tex`](paper/main.tex).

## Key Achievements

- **One-Command Reproducibility:** `python main.py` orchestrates data ingestion, model wrapping, circuit extraction, residual probing, and run-summary generation from the repository root.
- **Hooked Genomic Model Path:** Loads `zhihan1996/DNABERT-2-117M`, patches known environment compatibility issues, and falls back to a Hugging Face hook adapter when TransformerLens cannot fully wrap the custom BERT-style model.
- **Circuit-Level Exports:** Extracts selected-layer `W_Q`, `W_K`, `W_V`, and `W_O` tensors and saves exact `QK` and `OV` matrices under `results/circuits/`.
- **Probe-Ready Residual Caches:** Saves compact mean-pooled residual-stream arrays under `results/activations/` and trains logistic probes with AUROC, AUPRC, accuracy, example counts, and class balance.
- **Causal-Intervention Utilities:** Provides deterministic counterfactual motif mutation, restoration-metric computation, layer/head restoration matrix export, and heatmap generation.

## Overview

### What it does

MINTS turns genomic-transformer analysis into a reproducible pipeline. It prepares nucleotide sequence datasets, loads DNABERT-2, exports model internals, trains residual-stream probes, and writes a compact run summary that records what was analyzed and where the artifacts were saved.

### Why it matters

Genomic transformer predictions alone do not prove that a model has learned a biological mechanism. A high AUROC can come from shortcuts, dataset artifacts, or distributed representations that do not correspond to a clean motif detector. MINTS is designed to combine multiple kinds of evidence: linear decodability from residual streams, exact QK/OV circuit matrices, motif-local attention enrichment, and causal restoration under activation patching.

That combination matters because attention plots by themselves are not a proof. The goal is to build a pipeline where a candidate motif-sensitive head must survive stricter tests: it should attend to motif-support tokens, move information through an interpretable OV readout, make the biological feature linearly decodable, and restore the target signal when patched from a clean motif-containing run into a corrupted run.

### What is novel here

The repository is not just a dataset downloader or a classifier benchmark. Its focus is the bridge between computational biology and mechanistic interpretability: nucleotide tasks are treated as controlled biological probes, and transformer internals are exported in a form that can support circuit-level claims.

The practical novelty is the evidence stack. MINTS keeps the model fixed, freezes residual vectors for probing, computes exact attention-circuit matrices, records motif-destroying sequence edits, and exposes restoration metrics for causal intervention. The point is not to claim a motif detector from a single visualization; the point is to make the claim auditable.

### How it works

1. The pipeline reads configuration from `src/config.py` and creates the required `data/` and `results/` folders.
2. Hugging Face downstream nucleotide tasks are filtered, tokenized, and saved under `data/hf_downstream/`.
3. ENCODE CTCF artifacts are read from `data/ENCODE4_v1.5.1_GRCh38.txt` and stored under `data/encode/ctcf_gm12878/`.
4. GRCh38 sequence extraction creates CTCF sequence tables under `data/ctcf/`.
5. DNABERT-2 is loaded, placed on CUDA when available, and wrapped with TransformerLens or the local hook adapter.
6. Residual vectors are cached, QK/OV matrices are exported, and logistic probes are trained.
7. Counterfactual and patching utilities can generate motif-destroying pairs and save layer/head restoration tables and heatmaps.

### Why there are two datasets

The two data branches serve different roles:

- `hf_downstream` provides labeled promoter and splice-site classification tasks with train/test splits, making it the primary branch for residual probing and fast circuit experiments.
- `ctcf_gm12878` provides ENCODE-derived regulatory binding evidence, which is useful for moving from curated sequence benchmarks toward real genomic intervals.

### What we found

The latest completed run used all available rows in the configured train/test splits for residual caching and probing. It confirmed that the infrastructure, model loading, circuit export, residual caching, and probe training path works end to end on the local RTX 4060 setup.

- `promoter_tata`: AUROC `0.9137`, AUPRC `0.9241`, accuracy `0.8349`, `5062 / 212` train/test examples
- `promoter_no_tata`: AUROC `0.9383`, AUPRC `0.9475`, accuracy `0.8550`, `30000 / 1372` train/test examples
- `splice_sites_donors`: AUROC `0.8954`, AUPRC `0.9049`, accuracy `0.8230`, `30000 / 3000` train/test examples
- `splice_sites_acceptors`: AUROC `0.8847`, AUPRC `0.8954`, accuracy `0.8090`, `30000 / 3000` train/test examples

These numbers are a real positive result for the representation-level claim: the configured biological labels are linearly decodable from frozen layer-11 residual vectors. They do not prove that a specific attention head is a motif detector.

The important result so far is precise: MINTS achieved residual decodability and QK/OV export feasibility. It has not yet achieved the stronger mechanistic target, because QK-to-motif correlations, motif-local attention enrichment, and activation-patching restoration heatmaps still need to be executed and validated.

## Running

Create an environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the full repository pipeline:

```bash
python main.py
```

Run a capped debug pass:

```bash
python main.py --max-probe-train 512 --max-probe-test 256
```

`python main.py` is the full orchestration entrypoint. It uses all available probe rows by default, runs the configured pipeline sequentially, and writes a root run summary to `results/pipeline_run.json`.

Useful flags:

- `--overwrite`: rebuild generated datasets and redownload artifacts when needed
- `--max-probe-train`: cap train examples per task for activation caching and probing
- `--max-probe-test`: cap test examples per task for activation caching and probing
- `--json`: print a machine-readable completion payload

Notes:

- Omit `--max-probe-train` and `--max-probe-test` for full-data probing.
- CUDA is selected automatically when `torch.cuda.is_available()` is true.
- Large binary arrays under `results/activations/` and `results/circuits/` are generated artifacts and are ignored by Git.

## Data

The pipeline expects the ENCODE URL list in [`data/`](data):

- `ENCODE4_v1.5.1_GRCh38.txt`

The configured ENCODE URL file should include direct downloads for:

- `ENCFF680XUD.bigWig`
- `ENCFF827JRI.bed.gz`
- `ENCFF511URZ.bigBed`

The Hugging Face downstream data is downloaded programmatically and saved under:

- `data/hf_downstream/promoter_tata`
- `data/hf_downstream/promoter_no_tata`
- `data/hf_downstream/splice_sites_donors`
- `data/hf_downstream/splice_sites_acceptors`

CTCF-derived sequence tables are written under:

- `data/ctcf/`

## Pipeline Configuration

The default model and analysis parameters live in [`src/config.py`](src/config.py).

Core defaults:

- Model: `zhihan1996/DNABERT-2-117M`
- Hugging Face dataset: `InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`
- Tasks: `promoter_tata`, `promoter_no_tata`, `splice_sites_donors`, `splice_sites_acceptors`
- Activation layers: `0`, `5`, `11`
- Probe layer: `11`
- Circuit layers: `0`, `5`, `11`
- Batch size: `8`
- Seed: `1729`

Counterfactual mutation rules:

- TATA-like promoter motifs are replaced with a deterministic GC-balanced non-motif sequence.
- Splice donor `GT` dinucleotides are mutated to `GC`.
- Splice acceptor `AG` dinucleotides are mutated to `AC`.
- CTCF mutations use the local JASPAR CTCF matrix when available, with a deterministic core-motif fallback.

Patching rules:

- Restoration is computed as `(patched_logit - corrupted_logit) / (clean_logit - corrupted_logit)`.
- Degenerate pairs where `clean_logit == corrupted_logit` return `NaN`.
- Layer/head restoration matrices are saved as CSV files and heatmaps.

## Outputs

The pipeline writes a full artifact bundle under [`results/`](results):

- `results/pipeline_run.json`
- `results/tables/linear_probe_metrics.csv`
- `results/activations/`
- `results/circuits/`
- `results/counterfactuals/`
- `results/patching/`
- `results/figures/`
- `results/manifests/`

Top-level orchestration output:

- `results/pipeline_run.json`

Detailed manifests include:

- `hf_downstream_manifest.json`
- `encode_ctcf_gm12878_manifest.json`
- `grch38_manifest.json`
- `ctcf_sequences_manifest.json`
- `model_hooked_encoder_manifest.json`
- `circuits_manifest.json`
- `linear_probe_manifest.json`

## Latest Full Run

The current committed summary file is:

```bash
results/pipeline_run.json
```

That summary records the model, data sources, configured layers, artifact paths, and probe metrics for the latest completed local run.

Result bundles written by that run:

- [`results/pipeline_run.json`](results/pipeline_run.json)
- [`results/tables/linear_probe_metrics.csv`](results/tables/linear_probe_metrics.csv)
- `results/circuits/qk_ov_matrices.npz`

HF downstream full-run summary:

Role in the project: primary labeled benchmark branch for promoter and splice-site probing

- `promoter_tata`: `5062` train rows, `212` test rows
- `promoter_no_tata`: `30000` train rows, `1372` test rows
- `splice_sites_donors`: `30000` train rows, `3000` test rows
- `splice_sites_acceptors`: `30000` train rows, `3000` test rows
- Probe layer: `11`
- Runtime model backend: Hugging Face forward hooks
- Runtime device: `cuda`
- QK/OV export layers: `0`, `5`, `11`

Layer-11 residual probe metrics:

- `promoter_tata`: AUROC `0.9137`, AUPRC `0.9241`, accuracy `0.8349`
- `promoter_no_tata`: AUROC `0.9383`, AUPRC `0.9475`, accuracy `0.8550`
- `splice_sites_donors`: AUROC `0.8954`, AUPRC `0.9049`, accuracy `0.8230`
- `splice_sites_acceptors`: AUROC `0.8847`, AUPRC `0.8954`, accuracy `0.8090`

Theory comparison:

- Achieved: residual decodability above the paper's AUROC `0.80` threshold on all four tasks.
- Achieved: selected-layer QK/OV matrices were exported for downstream circuit tests.
- Not yet achieved: proof that any specific attention head is a biological motif detector.
- Still required: QK-to-motif correlation, motif-local attention enrichment, and causal restoration via activation patching.

Model instrumentation summary:

Role in the project: hooked nucleotide transformer backend for circuit and activation extraction

- Model: `zhihan1996/DNABERT-2-117M`
- Runtime device: CUDA when available
- Primary wrapper target: TransformerLens `HookedEncoder`
- Compatibility fallback: Hugging Face forward hooks
- Known limitation: DNABERT-2 under the fallback backend does not expose all TransformerLens-style per-head attention tensors

Intervention summary:

Role in the project: causal validation utilities for candidate motif detectors

- Counterfactual records preserve sequence length and mutation coordinates.
- Restoration matrices are shaped by layer and head.
- Heatmaps are written to `results/figures/`.
- TransformerLens head-output patching is used only when the active backend exposes the required patching utility.

## Repository Layout

- [`main.py`](main.py): CLI entry point for the one-command pipeline
- [`src/config.py`](src/config.py): paths, model defaults, task names, analysis layers, and run caps
- [`src/cli.py`](src/cli.py): command-line flags and pipeline invocation
- [`src/reproduce.py`](src/reproduce.py): orchestration and root run-summary writing
- [`src/data_ingestion.py`](src/data_ingestion.py): Hugging Face task filtering, tokenization, and ENCODE artifact handling
- [`src/ctcf.py`](src/ctcf.py): GRCh38 FASTA handling and CTCF sequence extraction
- [`src/modeling.py`](src/modeling.py): DNABERT-2 loading, compatibility patches, and hook adapter fallback
- [`src/activations.py`](src/activations.py): residual-stream caching for probe features
- [`src/circuits.py`](src/circuits.py): QK/OV matrix extraction
- [`src/probing.py`](src/probing.py): frozen residual logistic probes
- [`src/enrichment.py`](src/enrichment.py): motif-support attention enrichment utilities
- [`src/counterfactuals.py`](src/counterfactuals.py): motif-destroying clean/corrupted sequence pairs
- [`src/patching.py`](src/patching.py): restoration metrics, tensor patching, and heatmap export
- [`paper/main.pdf`](paper/main.pdf): compiled research paper
- [`paper/main.tex`](paper/main.tex): manuscript source
