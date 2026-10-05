"""Immutable stage receipts and fail-closed cache compatibility on resume."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .utils import sha256_file, utc_now_iso, write_json


def scientific_config(config):
    value = config.to_dict()
    # Hardware and output locations do not define a scientific estimand.
    value["model"].pop("device", None)
    value["model"].pop("local_files_only", None)
    return {"model": value["model"], "data": value["data"],
            "pooling": "attention-mask mean including special tokens", "schema": 1}


def config_fingerprint(config):
    return hashlib.sha256(json.dumps(scientific_config(config), sort_keys=True).encode()).hexdigest()


def _detail_files(value, project_root):
    files = set()
    if isinstance(value, dict):
        for item in value.values():
            files.update(_detail_files(item, project_root))
    elif isinstance(value, (list, tuple)):
        for item in value:
            files.update(_detail_files(item, project_root))
    elif isinstance(value, (str, Path)):
        try:
            path = Path(value)
            if not path.is_absolute():
                path = project_root / path
            if path.is_file():
                files.add(path.resolve())
        except (OSError, ValueError):
            pass
    return files


def stage_files(step, config, details):
    files = _detail_files(details, config.paths.project_root)
    if step == "ingest_hf_downstream":
        files.update(p.resolve() for p in config.paths.hf_downstream_dir.rglob("*") if p.is_file())
    elif step == "circuit_extraction_and_residual_probing":
        files.update(p.resolve() for p in config.paths.activations_dir.glob("*.npz"))
        files.update(p.resolve() for p in config.paths.circuits_dir.glob("*.npz"))
    elif step == "sequence_genomic_controls":
        files.update(p.resolve() for p in (config.paths.results_dir / "review").glob("ctcf_*") if p.is_file())
    elif step == "sequence_classifier_intervals":
        files.update(p.resolve() for p in (config.paths.results_dir / "review").glob("*_predictions.csv"))
    return files


def write_stage_receipt(step, config, details):
    files = stage_files(step, config, details)
    if not files:
        raise ValueError(f"Stage {step} produced no auditable artifacts")
    path = config.paths.manifests_dir / "lineage" / f"{step}.json"
    if path.exists():
        raise ValueError(f"Immutable stage receipt already exists: {step}; choose a fresh run")
    write_json(path, dict(stage=step, created_at=utc_now_iso(), fingerprint=config_fingerprint(config),
                         scientific_config=scientific_config(config),
                         artifacts={str(p): sha256_file(p) for p in sorted(files)},
                         source_sha256={p.relative_to(config.paths.project_root).as_posix(): sha256_file(p)
                                        for p in (config.paths.project_root / "src").glob("*.py")}))
    return path


def validate_resume(config, previous_steps):
    """Require receipts, matching configuration, unchanged artifacts and source."""
    expected = config_fingerprint(config)
    lineage = []
    for step in previous_steps:
        path = config.paths.manifests_dir / "lineage" / f"{step}.json"
        if not path.is_file():
            raise ValueError(f"Cannot resume without a verified stage receipt for {step}; start a fresh run")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        recorded = hashlib.sha256(json.dumps(receipt.get("scientific_config"), sort_keys=True).encode()).hexdigest()
        if receipt.get("fingerprint") != expected or recorded != expected or receipt.get("stage") != step:
            raise ValueError(f"Scientific configuration changed since {step}; checkpoint/tokenizer/settings are incompatible")
        for name, digest in receipt["artifacts"].items():
            artifact = Path(name)
            if not artifact.is_file() or sha256_file(artifact) != digest:
                raise ValueError(f"Upstream artifact changed or disappeared: {name}")
        for name, digest in receipt.get("source_sha256", {}).items():
            source = config.paths.project_root / name
            if not source.is_file() or sha256_file(source) != digest:
                raise ValueError(f"Scientific source changed since {step}: {name}; start a fresh run")
        lineage.append(dict(stage=step, receipt=str(path), receipt_sha256=sha256_file(path)))
    return lineage


def verified_import_manifest(paths, output, config, justification):
    """Record explicit reuse without assigning unknown historical checkpoint identity."""
    if not justification.strip():
        raise ValueError("Reused historical inputs require an explicit provenance statement")
    write_json(Path(output), dict(fingerprint=config_fingerprint(config), inputs={str(p): sha256_file(Path(p)) for p in paths},
                                 justification=justification, historical_checkpoint_identity="not inferred from artifact hashes"))


def verify_reproduced_caches(paths, task, comparison_path, reproduction_path, config):
    """Require cached bytes and pinned revisions to match the recorded clean run."""
    comparison = json.loads(Path(comparison_path).read_text(encoding="utf-8"))
    reproduction = json.loads(Path(reproduction_path).read_text(encoding="utf-8"))
    if reproduction.get("exit_status") != 0 or reproduction.get("model_revision") != config.model.revision or reproduction.get("dataset_revision") != config.data.hf_dataset_revision:
        raise ValueError("Cached clean reproduction uses incompatible checkpoint or dataset revisions")
    matches = [r for r in comparison.get("records", []) if r.get("task") == task]
    if len(matches) != 1:
        raise ValueError(f"Missing unique clean cache comparison for {task}")
    record = matches[0]
    if not all(record.get("identity", {}).get(k) is True for k in ("names","sequences","labels","layers")) or record.get("maximum_absolute_feature_difference") != 0:
        raise ValueError(f"Clean cache identity or feature equivalence failed for {task}")
    train, test = map(Path, paths)
    expected = record.get("training_cache_sha256", {}).get("new"), record.get("new_sha256")
    if expected[0] != record.get("training_cache_sha256", {}).get("old") or expected[1] != record.get("old_sha256"):
        raise ValueError(f"Historical and clean cache bytes differ for {task}")
    for path, digest in zip((train,test), expected):
        if not digest or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Cached features changed since the verified clean reproduction: {path}")
    return {"comparison_sha256":sha256_file(Path(comparison_path)),
            "reproduction_sha256":sha256_file(Path(reproduction_path))}
