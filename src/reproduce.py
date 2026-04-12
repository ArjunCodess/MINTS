"""One-command reproducibility orchestration for MINTS."""

from __future__ import annotations

import time
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import DEFAULT_CONFIG, PipelineConfig
from .ctcf import ensure_grch38_fasta, prepare_ctcf_sequences
from .activations import cache_probe_residuals
from .circuits import extract_qk_ov_matrices
from .data_ingestion import download_encode_artifacts, ingest_hf_downstream_tasks
from .modeling import load_hooked_encoder, summarize_hook_points
from .probing import run_all_probes
from .utils import utc_now_iso, write_json


@dataclass(frozen=True)
class StepRecord:
    """Execution record for a reproducibility step."""

    name: str
    status: str
    seconds: float
    details: dict[str, Any]


def _run_step(name: str, func: Callable[[], dict[str, Any]]) -> StepRecord:
    started = time.perf_counter()
    details = func()
    return StepRecord(
        name=name,
        status="completed",
        seconds=round(time.perf_counter() - started, 3),
        details=details,
    )


def _run_circuit_and_probe_exports(config: PipelineConfig) -> dict[str, Any]:
    """Load the model once and export circuit/probe artifacts."""

    bundle = load_hooked_encoder(config.model)
    hook_points = summarize_hook_points(bundle.hooked_model)
    model_manifest_path = config.paths.manifests_dir / "model_hooked_encoder_manifest.json"
    write_json(
        model_manifest_path,
        {
            "created_at": utc_now_iso(),
            "model_name": bundle.model_name,
            "device": bundle.device,
            "instrumentation_backend": bundle.instrumentation_backend,
            "instrumentation_error": bundle.instrumentation_error,
            "hook_point_count_reported": len(hook_points),
            "hook_points_sample": hook_points,
        },
    )

    activation_exports = cache_probe_residuals(bundle, config=config)
    circuit_export = extract_qk_ov_matrices(bundle, config=config)
    probe_table = run_all_probes(config=config)
    return {
        "model_manifest": str(model_manifest_path),
        "instrumentation_backend": bundle.instrumentation_backend,
        "model_name": bundle.model_name,
        "device": bundle.device,
        "activation_exports": [str(export.path) for export in activation_exports],
        "activation_summary": {
            "exports": len(activation_exports),
            "tasks": list(config.data.task_names),
            "splits": ["train", "test"],
            "layers": list(config.data.activation_layers),
            "pooling": "mean",
        },
        "circuit_export": str(circuit_export.path),
        "circuit_summary": {
            "layers": list(circuit_export.layers),
            "n_heads": circuit_export.n_heads,
            "d_model": circuit_export.d_model,
            "d_head": circuit_export.d_head,
        },
        "probe_table": str(probe_table),
    }


def _relative_to_project(path: str | Path, config: PipelineConfig) -> str:
    """Render a path relative to the repository root when possible."""

    path = Path(path)
    try:
        return str(path.relative_to(config.paths.project_root))
    except ValueError:
        return str(path)


def _load_probe_metrics(path: str | Path, config: PipelineConfig) -> list[dict[str, Any]]:
    """Load probe metrics for the top-level run manifest."""

    metrics_path = Path(path)
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "task": row["task"],
                    "layer": int(row["layer"]),
                    "auroc": float(row["auroc"]),
                    "auprc": float(row["auprc"]),
                    "accuracy": float(row["accuracy"]),
                    "train_examples": int(row["train_examples"]),
                    "test_examples": int(row["test_examples"]),
                }
            )
    return rows


def _compact_step(step: StepRecord, config: PipelineConfig) -> dict[str, Any]:
    """Keep pipeline_run.json readable while preserving useful run evidence."""

    details = step.details
    if step.name == "write_config":
        compact_details = {"path": _relative_to_project(details["path"], config)}
    elif step.name == "ingest_hf_downstream":
        compact_details = {
            "dataset": config.data.hf_dataset_name,
            "tasks": details["tasks"],
            "output_dir": _relative_to_project(config.paths.hf_downstream_dir, config),
        }
    elif step.name == "download_encode_ctcf":
        compact_details = {
            "artifacts": [
                {
                    "filename": Path(record["path"]).name,
                    "status": record["status"],
                    "path": _relative_to_project(record["path"], config),
                }
                for record in details["records"]
            ],
            "output_dir": _relative_to_project(config.paths.encode_dir, config),
        }
    elif step.name == "download_grch38":
        compact_details = {"path": _relative_to_project(details["path"], config)}
    elif step.name == "prepare_ctcf_sequences":
        compact_details = {"path": _relative_to_project(details["path"], config)}
    elif step.name == "circuit_extraction_and_residual_probing":
        compact_details = {
            "model": {
                "name": details["model_name"],
                "device": details["device"],
                "instrumentation_backend": details["instrumentation_backend"],
                "manifest": _relative_to_project(details["model_manifest"], config),
            },
            "activations": {
                **details["activation_summary"],
                "artifact_paths": [
                    _relative_to_project(path, config) for path in details["activation_exports"]
                ],
            },
            "circuits": {
                **details["circuit_summary"],
                "artifact_path": _relative_to_project(details["circuit_export"], config),
            },
            "probes": {
                "metrics_path": _relative_to_project(details["probe_table"], config),
                "metrics": _load_probe_metrics(details["probe_table"], config),
            },
        }
    else:
        compact_details = details

    return {
        "name": step.name,
        "status": step.status,
        "seconds": step.seconds,
        "details": compact_details,
    }


def _write_run_manifest(
    path: Path,
    config: PipelineConfig,
    status: str,
    overwrite: bool,
    steps: list[StepRecord],
    error: str | None = None,
) -> None:
    """Write the root run summary consumed by humans and future automation."""

    payload: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "status": status,
        "project": "MINTS",
        "overwrite": overwrite,
        "model": {
            "name": config.model.model_name,
            "device": config.model.device,
            "revision": config.model.revision,
            "trust_remote_code": config.model.trust_remote_code,
        },
        "data": {
            "hf_dataset": config.data.hf_dataset_name,
            "hf_dataset_config": config.data.hf_dataset_config,
            "tasks": list(config.data.task_names),
            "encode_url_file": _relative_to_project(config.paths.encode_url_file, config),
            "grch38_fasta_url": config.data.grch38_fasta_url,
        },
        "analysis": {
            "activation_layers": list(config.data.activation_layers),
            "probe_layer": config.data.probe_layer,
            "circuit_layers": list(config.data.circuit_layers),
            "max_probe_train": config.data.max_probe_train,
            "max_probe_test": config.data.max_probe_test,
        },
        "steps": [_compact_step(step, config) for step in steps],
    }
    if error is not None:
        payload["error"] = error
    write_json(path, payload)


def run_pipeline(
    config: PipelineConfig = DEFAULT_CONFIG,
    overwrite: bool = False,
) -> Path:
    """Run the complete reproducibility pipeline."""

    config.ensure_paths()
    run_manifest_path = config.paths.results_dir / "pipeline_run.json"
    steps: list[StepRecord] = []

    def record_config() -> dict[str, Any]:
        config_path = config.paths.manifests_dir / "pipeline_config.json"
        write_json(config_path, {"created_at": utc_now_iso(), "config": config.to_dict()})
        return {"path": str(config_path)}

    def ingest_hf() -> dict[str, Any]:
        summary = ingest_hf_downstream_tasks(
            config=config,
            overwrite=overwrite,
        )
        return {"tasks": summary}

    def download_encode() -> dict[str, Any]:
        records = download_encode_artifacts(
            config=config,
            overwrite=overwrite,
        )
        return {
            "records": [
                {"url": record.url, "path": str(record.output_path), "status": record.status}
                for record in records
            ],
        }

    def download_grch38() -> dict[str, Any]:
        path = ensure_grch38_fasta(config=config, overwrite=overwrite, decompress=True)
        return {"path": str(path)}

    def prepare_ctcf() -> dict[str, Any]:
        output_path = prepare_ctcf_sequences(
            config=config,
            flank=0,
        )
        return {"path": str(output_path)}

    try:
        steps.append(_run_step("write_config", record_config))
        steps.append(_run_step("ingest_hf_downstream", ingest_hf))
        steps.append(_run_step("download_encode_ctcf", download_encode))
        steps.append(_run_step("download_grch38", download_grch38))
        steps.append(_run_step("prepare_ctcf_sequences", prepare_ctcf))
        steps.append(_run_step("circuit_extraction_and_residual_probing", lambda: _run_circuit_and_probe_exports(config)))
    except Exception as exc:
        _write_run_manifest(
            run_manifest_path,
            config=config,
            status="failed",
            overwrite=overwrite,
            steps=steps,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    _write_run_manifest(
        run_manifest_path,
        config=config,
        status="completed",
        overwrite=overwrite,
        steps=steps,
    )
    return run_manifest_path
