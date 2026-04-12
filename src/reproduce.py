"""One-command reproducibility orchestration for MINTS."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .config import DEFAULT_CONFIG, PipelineConfig
from .ctcf import ensure_grch38_fasta, prepare_ctcf_sequences
from .data_ingestion import download_encode_artifacts, ingest_hf_downstream_tasks
from .modeling import load_hooked_encoder, summarize_hook_points
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


def _write_model_manifest(config: PipelineConfig) -> dict[str, Any]:
    bundle = load_hooked_encoder(config.model)
    hook_points = summarize_hook_points(bundle.hooked_model)
    manifest_path = config.paths.manifests_dir / "model_hooked_encoder_manifest.json"
    write_json(
        manifest_path,
        {
            "created_at": utc_now_iso(),
            "model_name": bundle.model_name,
            "device": bundle.device,
            "hook_point_count_reported": len(hook_points),
            "hook_points_sample": hook_points,
        },
    )
    return {
        "model_name": bundle.model_name,
        "device": bundle.device,
        "manifest": str(manifest_path),
        "hook_points_reported": len(hook_points),
    }


def run_pipeline(
    config: PipelineConfig = DEFAULT_CONFIG,
    overwrite: bool = False,
) -> Path:
    """Run the complete reproducibility pipeline."""

    config.ensure_paths()
    run_manifest_path = config.paths.manifests_dir / "pipeline_run.json"
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
        steps.append(_run_step("check_hooked_encoder", lambda: _write_model_manifest(config)))
    except Exception as exc:
        write_json(
            run_manifest_path,
            {
                "created_at": utc_now_iso(),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "config": config.to_dict(),
                "steps": [asdict(step) for step in steps],
            },
        )
        raise

    write_json(
        run_manifest_path,
        {
            "created_at": utc_now_iso(),
            "status": "completed",
            "config": config.to_dict(),
            "overwrite": overwrite,
            "steps": [asdict(step) for step in steps],
        },
    )
    return run_manifest_path
