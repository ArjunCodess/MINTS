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
from .distributed_features import run_distributed_feature_search
from .mechanistic_proofs import run_mechanistic_proof_exports, run_systematic_causal_intervention_exports
from .modeling import load_hooked_encoder, summarize_hook_points
from .probing import run_all_probes
from .utils import progress, utc_now_iso, write_json


PIPELINE_STEPS: tuple[str, ...] = (
    "write_config",
    "ingest_hf_downstream",
    "download_encode_ctcf",
    "download_grch38",
    "prepare_ctcf_sequences",
    "circuit_extraction_and_residual_probing",
    "strict_mechanistic_proofs",
    "systematic_causal_intervention",
    "distributed_feature_search",
)


@dataclass(frozen=True)
class StepRecord:
    """Execution record for a reproducibility step."""

    name: str
    status: str
    seconds: float
    details: dict[str, Any]


def _run_step(name: str, func: Callable[[], dict[str, Any]]) -> StepRecord:
    progress(f"Starting step: {name}")
    started = time.perf_counter()
    try:
        details = func()
    except Exception:
        progress(f"Failed step: {name}")
        raise
    seconds = round(time.perf_counter() - started, 3)
    progress(f"Completed step: {name} in {seconds:.3f}s")
    return StepRecord(
        name=name,
        status="completed",
        seconds=seconds,
        details=details,
    )


def _write_model_manifest(bundle: Any, config: PipelineConfig) -> Path:
    """Write model/hook metadata shared by circuit and proof runs."""

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
    return model_manifest_path


def _run_circuit_and_probe_exports(config: PipelineConfig) -> dict[str, Any]:
    """Load the model once and export circuit/probe artifacts."""

    progress(f"Loading model: {config.model.model_name}")
    bundle = load_hooked_encoder(config.model)
    progress(f"Model ready on {bundle.device} using {bundle.instrumentation_backend}")
    model_manifest_path = _write_model_manifest(bundle, config)

    progress("Caching residual activations for probe tasks")
    activation_exports = cache_probe_residuals(bundle, config=config)
    progress("Extracting QK/OV circuit matrices")
    circuit_export = extract_qk_ov_matrices(bundle, config=config)
    progress("Training linear probes from cached residuals")
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


def _qk_archive_has_low_rank_factors(config: PipelineConfig) -> bool:
    """Return whether the existing circuit archive contains W_Q/W_K factors."""

    import numpy as np

    archive_path = config.paths.circuits_dir / "qk_ov_matrices.npz"
    if not archive_path.exists():
        return False
    with np.load(archive_path) as archive:
        return "w_q" in archive.files and "w_k" in archive.files


def _run_strict_proof_exports(config: PipelineConfig) -> dict[str, Any]:
    """Load the model and run strict QK/enrichment/patching proof exports."""

    progress(f"Loading model for strict mechanistic proofs: {config.model.model_name}")
    bundle = load_hooked_encoder(config.model)
    progress(f"Model ready on {bundle.device} using {bundle.instrumentation_backend}")
    model_manifest_path = _write_model_manifest(bundle, config)

    if not _qk_archive_has_low_rank_factors(config):
        progress("Circuit archive is missing W_Q/W_K factors; refreshing QK/OV export before strict proofs")
        circuit_export = extract_qk_ov_matrices(bundle, config=config)
        circuit_summary: dict[str, Any] | None = {
            "artifact_path": str(circuit_export.path),
            "layers": list(circuit_export.layers),
            "n_heads": circuit_export.n_heads,
            "d_model": circuit_export.d_model,
            "d_head": circuit_export.d_head,
            "refreshed_for_low_rank_qk": True,
        }
    else:
        circuit_summary = None

    proof_exports = run_mechanistic_proof_exports(bundle, config=config)
    return {
        "model_manifest": str(model_manifest_path),
        "instrumentation_backend": bundle.instrumentation_backend,
        "model_name": bundle.model_name,
        "device": bundle.device,
        "circuit_refresh": circuit_summary,
        "proof_exports": proof_exports,
    }


def _run_systematic_causal_interventions(config: PipelineConfig) -> dict[str, Any]:
    """Load the model and run batch denoising activation patching."""

    progress(f"Loading model for systematic causal interventions: {config.model.model_name}")
    bundle = load_hooked_encoder(config.model)
    progress(f"Model ready on {bundle.device} using {bundle.instrumentation_backend}")
    model_manifest_path = _write_model_manifest(bundle, config)
    outputs = run_systematic_causal_intervention_exports(bundle, config=config)
    return {
        "model_manifest": str(model_manifest_path),
        "instrumentation_backend": bundle.instrumentation_backend,
        "model_name": bundle.model_name,
        "device": bundle.device,
        "systematic_interventions": outputs,
    }


def _run_distributed_feature_search(config: PipelineConfig) -> dict[str, Any]:
    """Load the model and run residual/MLP sparse feature search."""

    progress(f"Loading model for distributed feature search: {config.model.model_name}")
    bundle = load_hooked_encoder(config.model)
    progress(f"Model ready on {bundle.device} using {bundle.instrumentation_backend}")
    model_manifest_path = _write_model_manifest(bundle, config)
    outputs = run_distributed_feature_search(
        bundle,
        config=config,
        max_sequences=config.data.max_feature_search_sequences,
    )
    return {
        "model_manifest": str(model_manifest_path),
        "instrumentation_backend": bundle.instrumentation_backend,
        "model_name": bundle.model_name,
        "device": bundle.device,
        "distributed_features": outputs,
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
                    "train_positive_rate": float(row["train_positive_rate"])
                    if "train_positive_rate" in row
                    else None,
                    "test_positive_rate": float(row["test_positive_rate"])
                    if "test_positive_rate" in row
                    else None,
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
    elif step.name == "strict_mechanistic_proofs":
        compact_details = {
            "model": {
                "name": details["model_name"],
                "device": details["device"],
                "instrumentation_backend": details["instrumentation_backend"],
                "manifest": _relative_to_project(details["model_manifest"], config),
            },
            "circuit_refresh": (
                {
                    **details["circuit_refresh"],
                    "artifact_path": _relative_to_project(
                        details["circuit_refresh"]["artifact_path"],
                        config,
                    ),
                }
                if details.get("circuit_refresh") is not None
                else None
            ),
            "strict_proofs": {
                "qk_alignment": {
                    key: _relative_to_project(path, config)
                    for key, path in details["proof_exports"]["qk_alignment"].items()
                },
                "activation_patching": {
                    key: _relative_to_project(path, config)
                    for key, path in details["proof_exports"]["activation_patching"].items()
                },
                "patching_pair": _relative_to_project(details["proof_exports"]["patching_pair"], config),
                "patching_pair_summary": details["proof_exports"]["patching_pair_summary"],
                "summary": details["proof_exports"].get("summary", {}),
            },
        }
    elif step.name == "systematic_causal_intervention":
        compact_details = {
            "model": {
                "name": details["model_name"],
                "device": details["device"],
                "instrumentation_backend": details["instrumentation_backend"],
                "manifest": _relative_to_project(details["model_manifest"], config),
            },
            "tasks": {
                task: {
                    key: (
                        _relative_to_project(value, config)
                        if key in {"table", "figure", "manifest", "pairs"}
                        else value
                    )
                    for key, value in outputs.items()
                }
                for task, outputs in details["systematic_interventions"].items()
            },
        }
    elif step.name == "distributed_feature_search":
        outputs = details["distributed_features"]
        compact_details = {
            "model": {
                "name": details["model_name"],
                "device": details["device"],
                "instrumentation_backend": details["instrumentation_backend"],
                "manifest": _relative_to_project(details["model_manifest"], config),
            },
            "activation_path": _relative_to_project(outputs["activation_path"], config),
            "summary_path": _relative_to_project(outputs["summary_path"], config),
            "combined_alignment_path": _relative_to_project(outputs["combined_alignment_path"], config),
            "sources": {
                source: {
                    key: (
                        _relative_to_project(value, config)
                        if key in {"checkpoint", "history", "alignment", "figure"}
                        else value
                    )
                    for key, value in source_outputs.items()
                }
                for source, source_outputs in outputs["sources"].items()
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
    from_step: str,
    steps: list[StepRecord],
    error: str | None = None,
) -> None:
    """Write the root run summary consumed by humans and future automation."""

    payload: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "status": status,
        "project": "MINTS",
        "overwrite": overwrite,
        "from_step": from_step,
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
            "max_qk_alignment_sequences": config.data.max_qk_alignment_sequences,
            "max_patching_pairs": config.data.max_patching_pairs,
            "max_feature_search_sequences": config.data.max_feature_search_sequences,
            "sae_dictionary_size": config.data.sae_dictionary_size,
            "sae_epochs": config.data.sae_epochs,
            "sae_l1_coefficient": config.data.sae_l1_coefficient,
        },
        "steps": [_compact_step(step, config) for step in steps],
    }
    if error is not None:
        payload["error"] = error
    write_json(path, payload)


def run_pipeline(
    config: PipelineConfig = DEFAULT_CONFIG,
    overwrite: bool = False,
    from_step: str = "write_config",
) -> Path:
    """Run the complete reproducibility pipeline."""

    if from_step not in PIPELINE_STEPS:
        raise ValueError(f"Unknown from_step '{from_step}'. Expected one of: {', '.join(PIPELINE_STEPS)}")
    progress("Creating required data/results directories")
    config.ensure_paths()
    run_manifest_path = (
        config.paths.results_dir / "pipeline_run.json"
        if from_step == "write_config"
        else config.paths.results_dir / f"pipeline_run_{from_step}.json"
    )
    steps: list[StepRecord] = []

    def record_config() -> dict[str, Any]:
        config_path = config.paths.manifests_dir / "pipeline_config.json"
        progress(f"Writing config snapshot to {config_path}")
        write_json(config_path, {"created_at": utc_now_iso(), "config": config.to_dict()})
        return {"path": str(config_path)}

    def ingest_hf() -> dict[str, Any]:
        progress("Preparing Hugging Face downstream task datasets")
        summary = ingest_hf_downstream_tasks(
            config=config,
            overwrite=overwrite,
        )
        return {"tasks": summary}

    def download_encode() -> dict[str, Any]:
        progress("Checking ENCODE CTCF artifacts")
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
        progress("Checking GRCh38 FASTA")
        path = ensure_grch38_fasta(config=config, overwrite=overwrite, decompress=True)
        return {"path": str(path)}

    def prepare_ctcf() -> dict[str, Any]:
        progress("Preparing CTCF peak sequence table")
        output_path = prepare_ctcf_sequences(
            config=config,
            flank=0,
        )
        return {"path": str(output_path)}

    try:
        step_functions: dict[str, Callable[[], dict[str, Any]]] = {
            "write_config": record_config,
            "ingest_hf_downstream": ingest_hf,
            "download_encode_ctcf": download_encode,
            "download_grch38": download_grch38,
            "prepare_ctcf_sequences": prepare_ctcf,
            "circuit_extraction_and_residual_probing": lambda: _run_circuit_and_probe_exports(config),
            "strict_mechanistic_proofs": lambda: _run_strict_proof_exports(config),
            "systematic_causal_intervention": lambda: _run_systematic_causal_interventions(config),
            "distributed_feature_search": lambda: _run_distributed_feature_search(config),
        }
        start_index = PIPELINE_STEPS.index(from_step)
        for step_name in PIPELINE_STEPS[start_index:]:
            steps.append(_run_step(step_name, step_functions[step_name]))
    except Exception as exc:
        progress(f"Writing failed run manifest to {run_manifest_path}")
        _write_run_manifest(
            run_manifest_path,
            config=config,
            status="failed",
            overwrite=overwrite,
            from_step=from_step,
            steps=steps,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    progress(f"Writing completed run manifest to {run_manifest_path}")
    _write_run_manifest(
        run_manifest_path,
        config=config,
        status="completed",
        overwrite=overwrite,
        from_step=from_step,
        steps=steps,
    )
    return run_manifest_path
