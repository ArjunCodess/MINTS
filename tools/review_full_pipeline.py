"""Uncapped full pipeline in a fresh data/result tree; never overwrite June data."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import argparse
import importlib.metadata
import json
import platform
import time
from src.config import DEFAULT_CONFIG, ProjectPaths
from src.reproduce import run_pipeline, PIPELINE_STEPS
from src.utils import sha256_file

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-directory",default="results/review/full_run_retry")
parser.add_argument("--from-step",choices=PIPELINE_STEPS,default="write_config")
parser.add_argument("--blas-threads",type=int,default=1)
args=parser.parse_args()
if args.blas_threads<1:
    parser.error("--blas-threads must be positive")
base = (DEFAULT_CONFIG.paths.project_root / args.run_directory).resolve()
if not Path(args.run_directory).is_absolute() and not base.is_relative_to(DEFAULT_CONFIG.paths.project_root / "results/review"):
    raise ValueError("Relative isolated runs must stay under results/review; an absolute path may select another disk")
manifest_name="pipeline_run.json" if args.from_step=="write_config" else f"pipeline_run_{args.from_step}.json"
if (base / "results" / manifest_name).exists():
    raise RuntimeError("Full-run directory already has a run; preserve it and choose a new directory")
data, results = base / "data", base / "results"
paths = ProjectPaths(project_root=DEFAULT_CONFIG.paths.project_root, data_dir=data, results_dir=results,
    hf_downstream_dir=data/"hf_downstream", encode_dir=data/"encode/ctcf_gm12878", ctcf_dir=data/"ctcf",
    manifests_dir=results/"manifests", activations_dir=results/"activations", circuits_dir=results/"circuits",
    enrichment_dir=results/"enrichment", qk_alignment_dir=results/"qk_alignment", counterfactuals_dir=results/"counterfactuals",
    patching_dir=results/"patching", distributed_features_dir=results/"distributed_features",
    cross_model_dir=results/"cross_model", figures_dir=results/"figures", tables_dir=results/"tables",
    encode_url_file=DEFAULT_CONFIG.paths.encode_url_file, grch38_fasta_gz=data/"genomes/hg38.fa.gz", grch38_fasta=data/"genomes/hg38.fa")
config=replace(DEFAULT_CONFIG, paths=paths)
started=time.perf_counter()
base.mkdir(parents=True,exist_ok=True)
environment=dict(python=platform.python_version(),executable=sys.executable,
                 packages={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
                 command=[sys.executable,*sys.argv],seed=config.data.seed)
environment["blas_threads"]=args.blas_threads
environment["source_sha256"]={str(p.relative_to(DEFAULT_CONFIG.paths.project_root)):sha256_file(p)
                              for folder in ["src","tools"] for p in (DEFAULT_CONFIG.paths.project_root/folder).glob("*.py")}
environment_path=base/f"environment_{args.from_step}.json"
if environment_path.exists():
    environment_path=base/f"environment_{args.from_step}_{time.time_ns()}.json"
environment_path.write_text(json.dumps(environment,indent=2),encoding="utf-8")
status="failed"
try:
    # Initialize native libraries before changing their thread pools.
    import torch
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.blas_threads,user_api="blas"):
        run_pipeline(config,from_step=args.from_step)
    status="completed"
finally:
    artifacts={str(p.relative_to(base)):sha256_file(p) for p in results.rglob("*") if p.is_file()}
    (base/f"artifacts_{args.from_step}.json").write_text(json.dumps(
        dict(status=status,seconds=time.perf_counter()-started,sha256=artifacts),indent=2),encoding="utf-8")
