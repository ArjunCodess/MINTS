"""Cross-model tokenization comparison for nucleotide transformers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .activations import cache_probe_residuals
from .circuits import extract_qk_ov_matrices
from .config import DEFAULT_CONFIG, PipelineConfig, ProjectPaths
from .modeling import (
    DNABERT2_MODEL_NAME,
    NUCLEOTIDE_TRANSFORMER_MODEL_NAME,
    load_hooked_encoder,
    model_slug,
    tokenization_family,
)
from .probing import run_all_probes
from .qk_alignment import run_ctcf_qk_alignment
from .utils import progress, utc_now_iso, write_json


CROSS_MODEL_NAMES: tuple[str, str] = (
    DNABERT2_MODEL_NAME,
    NUCLEOTIDE_TRANSFORMER_MODEL_NAME,
)


def cross_model_paths(config: PipelineConfig, slug: str) -> ProjectPaths:
    """Return isolated result paths for one comparison model."""

    root = config.paths.cross_model_dir / slug
    return ProjectPaths(
        project_root=config.paths.project_root,
        data_dir=config.paths.data_dir,
        results_dir=root,
        hf_downstream_dir=config.paths.hf_downstream_dir,
        encode_dir=config.paths.encode_dir,
        ctcf_dir=config.paths.ctcf_dir,
        manifests_dir=root / "manifests",
        activations_dir=root / "activations",
        circuits_dir=root / "circuits",
        enrichment_dir=root / "enrichment",
        qk_alignment_dir=root / "qk_alignment",
        counterfactuals_dir=root / "counterfactuals",
        patching_dir=root / "patching",
        distributed_features_dir=root / "distributed_features",
        cross_model_dir=config.paths.cross_model_dir,
        figures_dir=root / "figures",
        tables_dir=root / "tables",
        encode_url_file=config.paths.encode_url_file,
        grch38_fasta_gz=config.paths.grch38_fasta_gz,
        grch38_fasta=config.paths.grch38_fasta,
    )


def build_cross_model_config(config: PipelineConfig, model_name: str) -> PipelineConfig:
    """Create a model-specific config while sharing all input data."""

    return replace(
        config,
        model=replace(config.model, model_name=model_name),
        paths=cross_model_paths(config, model_slug(model_name)),
    )


def _read_probe_metrics(path: Path) -> list[dict[str, Any]]:
    """Load linear-probe metrics into JSON-friendly records."""

    table = pd.read_csv(path)
    records: list[dict[str, Any]] = []
    for row in table.itertuples(index=False):
        records.append(
            {
                "task": str(row.task),
                "layer": int(row.layer),
                "auroc": float(row.auroc),
                "auprc": float(row.auprc),
                "accuracy": float(row.accuracy),
                "train_examples": int(row.train_examples),
                "test_examples": int(row.test_examples),
            }
        )
    return records


def summarize_attention_enrichment(path: Path | None) -> dict[str, Any]:
    """Summarize a matched attention-enrichment table."""

    if path is None or not path.exists():
        return {
            "status": "skipped",
            "best_rho": None,
            "candidate_heads": 0,
            "top_heads": [],
        }
    table = pd.read_csv(path)
    if "rho" not in table.columns:
        return {
            "status": "missing_rho",
            "best_rho": None,
            "candidate_heads": 0,
            "top_heads": [],
        }
    finite = table[np.isfinite(table["rho"])]
    if finite.empty:
        return {
            "status": "no_finite_scores",
            "best_rho": None,
            "candidate_heads": 0,
            "top_heads": [],
        }
    if "passes_attention_enrichment" in finite.columns:
        candidates = finite[finite["passes_attention_enrichment"].astype(bool)]
    else:
        candidates = finite[finite["rho"] >= 2.0]
    top = finite.sort_values("rho", ascending=False).head(10)
    return {
        "status": "completed",
        "best_rho": float(finite["rho"].max()),
        "mean_rho": float(finite["rho"].mean()),
        "candidate_heads": int(len(candidates)),
        "top_heads": [
            {
                "layer": int(row.layer),
                "head": int(row.head),
                "rho": float(row.rho),
            }
            for row in top.itertuples(index=False)
        ],
    }


def probe_metric_deltas(
    base_metrics: list[dict[str, Any]],
    comparison_metrics: list[dict[str, Any]],
    base_name: str = DNABERT2_MODEL_NAME,
    comparison_name: str = NUCLEOTIDE_TRANSFORMER_MODEL_NAME,
) -> list[dict[str, Any]]:
    """Compute per-task metric deltas for the comparison report."""

    base_by_task = {row["task"]: row for row in base_metrics}
    comparison_by_task = {row["task"]: row for row in comparison_metrics}
    rows: list[dict[str, Any]] = []
    for task in sorted(set(base_by_task) & set(comparison_by_task)):
        for metric in ("auroc", "auprc", "accuracy"):
            base_value = float(base_by_task[task][metric])
            comparison_value = float(comparison_by_task[task][metric])
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "base_model": base_name,
                    "comparison_model": comparison_name,
                    "base_value": base_value,
                    "comparison_value": comparison_value,
                    "delta_comparison_minus_base": comparison_value - base_value,
                }
            )
    return rows


def attention_enrichment_delta(
    base_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
    base_name: str = DNABERT2_MODEL_NAME,
    comparison_name: str = NUCLEOTIDE_TRANSFORMER_MODEL_NAME,
) -> dict[str, Any]:
    """Compute cross-model attention enrichment differences."""

    base_rho = base_summary.get("best_rho")
    comparison_rho = comparison_summary.get("best_rho")
    return {
        "base_model": base_name,
        "comparison_model": comparison_name,
        "base_best_rho": base_rho,
        "comparison_best_rho": comparison_rho,
        "delta_best_rho_comparison_minus_base": (
            float(comparison_rho) - float(base_rho)
            if base_rho is not None and comparison_rho is not None
            else None
        ),
        "base_candidate_heads": int(base_summary.get("candidate_heads", 0)),
        "comparison_candidate_heads": int(comparison_summary.get("candidate_heads", 0)),
        "delta_candidate_heads_comparison_minus_base": int(comparison_summary.get("candidate_heads", 0))
        - int(base_summary.get("candidate_heads", 0)),
    }


def _run_single_model_comparison(
    model_name: str,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Run probe and attention-enrichment benchmarks for one model."""

    model_config = build_cross_model_config(config, model_name)
    model_config.ensure_paths()
    progress(f"Cross-model comparison loading {model_name}")
    bundle = load_hooked_encoder(model_config.model)
    family = tokenization_family(model_name, bundle.tokenizer)
    progress(f"Cross-model benchmark ready: model={model_name}, tokenization={family}, device={bundle.device}")
    cache_probe_residuals(bundle, config=model_config)
    probe_table = run_all_probes(config=model_config)

    attention_summary: dict[str, Any]
    enrichment_table: Path | None = None
    circuit_path: Path | None = None
    qk_outputs: dict[str, Path] = {}
    skip_reason: str | None = None
    try:
        circuit_export = extract_qk_ov_matrices(bundle, config=model_config)
        circuit_path = circuit_export.path
        qk_outputs = run_ctcf_qk_alignment(
            bundle,
            config=model_config,
            max_sequences=config.data.max_cross_model_qk_alignment_sequences,
            output_stem=f"{model_slug(model_name)}_ctcf_qk_alignment",
        )
        enrichment_table = qk_outputs.get("enrichment_table")
        attention_summary = summarize_attention_enrichment(enrichment_table)
    except Exception as exc:
        skip_reason = f"{type(exc).__name__}: {exc}"
        attention_summary = {
            "status": "skipped",
            "skip_reason": skip_reason,
            "best_rho": None,
            "candidate_heads": 0,
            "top_heads": [],
        }
        progress(f"Skipping attention-enrichment comparison for {model_name}: {skip_reason}")

    probe_metrics = _read_probe_metrics(probe_table)
    return {
        "model_name": model_name,
        "model_slug": model_slug(model_name),
        "tokenization_family": family,
        "tokenizer_class": type(bundle.tokenizer).__name__,
        "device": bundle.device,
        "instrumentation_backend": bundle.instrumentation_backend,
        "instrumentation_error": bundle.instrumentation_error,
        "result_root": str(model_config.paths.results_dir),
        "probe_metrics_path": str(probe_table),
        "probe_metrics": probe_metrics,
        "circuit_path": str(circuit_path) if circuit_path is not None else None,
        "qk_alignment_outputs": {key: str(value) for key, value in qk_outputs.items()},
        "attention_enrichment": attention_summary,
    }


def run_cross_model_tokenization_comparison(
    config: PipelineConfig = DEFAULT_CONFIG,
    model_names: tuple[str, str] = CROSS_MODEL_NAMES,
) -> Path:
    """Run DNABERT-2 vs Nucleotide Transformer probing/enrichment comparison."""

    config.ensure_paths()
    results = [_run_single_model_comparison(model_name, config=config) for model_name in model_names]
    by_model = {record["model_name"]: record for record in results}
    base = by_model.get(DNABERT2_MODEL_NAME, results[0])
    comparison = by_model.get(NUCLEOTIDE_TRANSFORMER_MODEL_NAME, results[-1])
    report = {
        "created_at": utc_now_iso(),
        "models": by_model,
        "probe_metric_deltas": probe_metric_deltas(
            base["probe_metrics"],
            comparison["probe_metrics"],
            base_name=base["model_name"],
            comparison_name=comparison["model_name"],
        ),
        "attention_enrichment_delta": attention_enrichment_delta(
            base["attention_enrichment"],
            comparison["attention_enrichment"],
            base_name=base["model_name"],
            comparison_name=comparison["model_name"],
        ),
    }
    output_path = config.paths.tables_dir / "cross_model_tokenization_comparison.json"
    write_json(output_path, report)
    progress(f"Writing cross-model tokenization comparison report: {output_path}")
    return output_path
