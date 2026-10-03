"""Uncapped full pipeline in a fresh data/result tree; never overwrite June data."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
from src.config import DEFAULT_CONFIG, ProjectPaths
from src.reproduce import run_pipeline

base = DEFAULT_CONFIG.paths.project_root / "results/review/full_run"
if (base / "results/pipeline_run.json").exists():
    raise RuntimeError("Full-run directory already has a run; preserve it and choose a new directory")
data, results = base / "data", base / "results"
paths = ProjectPaths(project_root=DEFAULT_CONFIG.paths.project_root, data_dir=data, results_dir=results,
    hf_downstream_dir=data/"hf_downstream", encode_dir=data/"encode/ctcf_gm12878", ctcf_dir=data/"ctcf",
    manifests_dir=results/"manifests", activations_dir=results/"activations", circuits_dir=results/"circuits",
    enrichment_dir=results/"enrichment", qk_alignment_dir=results/"qk_alignment", counterfactuals_dir=results/"counterfactuals",
    patching_dir=results/"patching", distributed_features_dir=results/"distributed_features",
    cross_model_dir=results/"cross_model", figures_dir=results/"figures", tables_dir=results/"tables",
    encode_url_file=DEFAULT_CONFIG.paths.encode_url_file, grch38_fasta_gz=data/"genomes/hg38.fa.gz", grch38_fasta=data/"genomes/hg38.fa")
run_pipeline(replace(DEFAULT_CONFIG, paths=paths))
