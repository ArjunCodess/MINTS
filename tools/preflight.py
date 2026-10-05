"""Report supported runtime, cached scientific inputs, and disk headroom."""
from pathlib import Path
import argparse
import importlib.metadata as metadata
import json
import platform
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DEFAULT_CONFIG

def inspect_environment(require_scientific=False):
    import torch
    root = DEFAULT_CONFIG.paths.project_root
    versions, errors = {}, []
    for line in (root / "requirements.txt").read_text().splitlines():
        if "==" not in line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        try:
            actual = metadata.version(name)
            versions[name] = actual
            if actual.split("+")[0] != expected:
                errors.append(f"{name}: expected {expected}, found {actual}")
        except metadata.PackageNotFoundError:
            errors.append(f"{name}: missing")
    if platform.python_version_tuple()[:2] != ("3", "14"):
        errors.append("The tested runtime is Python 3.14")
    paths = DEFAULT_CONFIG.paths
    inputs = {
        "ctcf_sequences": paths.ctcf_dir / "ctcf_gm12878_sequences.tsv",
        "reference_genome": paths.grch38_fasta,
        **{f"{task}_{split}": paths.activations_dir / f"{task}_{split}_residual_mean.npz"
           for task in DEFAULT_CONFIG.data.task_names for split in ("train", "test")},
    }
    present = {name: path.exists() for name, path in inputs.items()}
    tokenizer_cached = False
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(DEFAULT_CONFIG.model.model_name,
            revision=DEFAULT_CONFIG.model.revision, trust_remote_code=True, local_files_only=True)
        tokenizer_cached = True
    except (OSError, ValueError):
        pass
    if require_scientific:
        errors.extend(f"Missing scientific input: {name}" for name, available in present.items() if not available)
        if not tokenizer_cached:
            errors.append("Pinned tokenizer is not cached; ingest the pinned checkpoint first")
        if shutil.disk_usage(root).free < 512 * 1024**2:
            errors.append("Less than 512 MiB free for isolated outputs")
    return dict(python=platform.python_version(), packages=versions,
        cuda_available=torch.cuda.is_available(),
        cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        disk_free_bytes=shutil.disk_usage(root).free, scientific_inputs=present,
        pinned_tokenizer_cached=tokenizer_cached, errors=errors,
        scope="CPU tests use synthetic fixtures; scientific execution requires cached data/model and network for the pinned DNase file")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scientific", action="store_true")
    args = parser.parse_args()
    report = inspect_environment(args.scientific)
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["errors"]))
