"""Strict mechanistic proof orchestration for motif-detector claims."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .counterfactuals import MutationRecord, generate_counterfactual_sequence, save_counterfactual_pairs
from .patching import run_batch_dnabert_activation_patching, run_custom_dnabert_activation_patching
from .qk_alignment import run_ctcf_qk_alignment
from .utils import progress


def _candidate_records(path: str | Path, flag_column: str, metric_columns: tuple[str, ...]) -> list[dict[str, Any]]:
    """Read candidate head rows from an artifact table for the root run summary."""

    table_path = Path(path)
    if not str(path) or not table_path.exists() or not table_path.is_file():
        return []
    table = pd.read_csv(table_path)
    if flag_column not in table.columns:
        return []
    rows: list[dict[str, Any]] = []
    for row in table[table[flag_column]].itertuples(index=False):
        row_dict = row._asdict()
        record: dict[str, Any] = {
            "layer": int(row_dict["layer"]),
            "head": int(row_dict["head"]),
        }
        for column in metric_columns:
            value = row_dict.get(column)
            if value is not None:
                record[column] = float(value)
        rows.append(record)
    return rows


def _patching_summary(path: str | Path) -> dict[str, Any]:
    """Summarize activation-patching restoration scores for pipeline_run.json."""

    table_path = Path(path)
    if not table_path.exists():
        return {}
    table = pd.read_csv(table_path)
    if "restoration" not in table.columns or table.empty:
        return {}
    finite = table["restoration"].dropna()
    if finite.empty:
        return {"finite_scores": 0, "max_restoration": None, "top_heads": []}
    top = table.dropna(subset=["restoration"]).sort_values("restoration", ascending=False).head(10)
    return {
        "finite_scores": int(len(finite)),
        "max_restoration": float(finite.max()),
        "mean_restoration": float(finite.mean()),
        "top_heads": [
            {
                "layer": int(row.layer),
                "head": int(row.head),
                "restoration": float(row.restoration),
            }
            for row in top.itertuples(index=False)
        ],
    }


def _token_shapes_match(bundle: Any, clean_sequence: str, corrupted_sequence: str) -> bool:
    """Return whether a counterfactual pair can be patched position-wise."""

    clean = bundle.tokenizer(clean_sequence, return_tensors="pt", padding=True, truncation=True)
    corrupted = bundle.tokenizer(corrupted_sequence, return_tensors="pt", padding=True, truncation=True)
    return clean["input_ids"].shape == corrupted["input_ids"].shape


def _select_patching_pair(
    bundle: Any,
    config: PipelineConfig,
    task: str = "promoter_tata",
) -> MutationRecord:
    """Select a positive example whose counterfactual preserves token shape."""

    from datasets import load_from_disk

    dataset_path = config.paths.hf_downstream_dir / task
    if not dataset_path.exists():
        raise FileNotFoundError(f"Task dataset not found for activation patching: {dataset_path}")
    dataset = load_from_disk(str(dataset_path))
    for split_name in ("test", "train"):
        if split_name not in dataset:
            continue
        split = dataset[split_name]
        for row_idx, row in enumerate(split):
            if int(row["label"]) != 1:
                continue
            try:
                record = generate_counterfactual_sequence(
                    sequence=str(row["sequence"]),
                    task=task,
                    sequence_id=str(row.get("name", f"{split_name}_{row_idx}")),
                    allow_center_fallback=False,
                    config=config,
                )
            except ValueError:
                continue
            if _token_shapes_match(bundle, record.clean_sequence, record.corrupted_sequence):
                return record
    raise ValueError(
        f"Could not find a positive {task} sequence whose motif-destruction counterfactual "
        "preserves DNABERT token shape."
    )


def run_mechanistic_proof_exports(
    bundle: Any,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Run strict proof exports using existing model, residual, and circuit artifacts."""

    progress("Running CTCF motif scoring, QK alignment, and matched attention enrichment")
    qk_outputs = run_ctcf_qk_alignment(
        bundle=bundle,
        config=config,
        max_sequences=config.data.max_qk_alignment_sequences,
        output_stem="ctcf_qk_alignment",
    )

    progress("Selecting a clean/corrupted promoter-TATA pair for custom activation patching")
    patching_pair = _select_patching_pair(bundle=bundle, config=config, task="promoter_tata")
    pair_path = save_counterfactual_pairs(
        [patching_pair],
        "promoter_tata_activation_patching_pair.tsv",
        config=config,
    )

    progress("Running DNABERT forward-hook activation patching across all layers and heads")
    patching_outputs = run_custom_dnabert_activation_patching(
        bundle=bundle,
        clean_sequence=patching_pair.clean_sequence,
        corrupted_sequence=patching_pair.corrupted_sequence,
        task="promoter_tata",
        config=config,
        layer_indices=None,
        output_stem="promoter_tata_dnabert_activation_patching",
    )
    qk_candidates = _candidate_records(
        qk_outputs["table"],
        flag_column="passes_qk_alignment",
        metric_columns=("pearson_r", "p_value", "n_positions"),
    )
    enrichment_candidates = _candidate_records(
        qk_outputs.get("enrichment_table", ""),
        flag_column="passes_attention_enrichment",
        metric_columns=("rho", "a_motif", "a_bg", "support_tokens"),
    )
    return {
        "qk_alignment": {key: str(value) for key, value in qk_outputs.items()},
        "activation_patching": {key: str(value) for key, value in patching_outputs.items()},
        "patching_pair": str(Path(pair_path)),
        "patching_pair_summary": {
            "task": patching_pair.task,
            "sequence_id": patching_pair.sequence_id,
            "motif_name": patching_pair.motif_name,
            "start": patching_pair.start,
            "end": patching_pair.end,
            "clean_subsequence": patching_pair.clean_subsequence,
            "corrupted_subsequence": patching_pair.corrupted_subsequence,
        },
        "summary": {
            "qk_alignment_candidate_count": len(qk_candidates),
            "qk_alignment_candidates": qk_candidates,
            "attention_enrichment_candidate_count": len(enrichment_candidates),
            "attention_enrichment_candidates": enrichment_candidates,
            "activation_patching": _patching_summary(patching_outputs["table"]),
        },
    }


def _batch_patching_summary(path: str | Path) -> dict[str, Any]:
    """Summarize a systematic batch-patching table."""

    table_path = Path(path)
    if not table_path.exists():
        return {}
    table = pd.read_csv(table_path)
    if "restoration" not in table.columns:
        return {}
    finite = table.dropna(subset=["restoration"])
    if finite.empty:
        return {"finite_scores": 0, "max_restoration": None, "top_heads": []}
    top = finite.sort_values("restoration", ascending=False).head(10)
    return {
        "finite_scores": int(len(finite)),
        "max_restoration": float(finite["restoration"].max()),
        "mean_restoration": float(finite["restoration"].mean()),
        "pairs": int(finite["pairs"].iloc[0]) if "pairs" in finite.columns else None,
        "denominator_failures": int(finite["denominator_failures"].iloc[0])
        if "denominator_failures" in finite.columns
        else None,
        "top_heads": [
            {
                "layer": int(row.layer),
                "head": int(row.head),
                "restoration": float(row.restoration),
            }
            for row in top.itertuples(index=False)
        ],
    }


def run_systematic_causal_intervention_exports(
    bundle: Any,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Run batch denoising activation patching for the configured causal tasks."""

    tasks = ("promoter_tata", "splice_sites_donors")
    outputs: dict[str, Any] = {}
    for task in tasks:
        progress(f"Running systematic denoising activation patching for {task}")
        task_outputs = run_batch_dnabert_activation_patching(
            bundle=bundle,
            task=task,
            config=config,
            max_pairs=config.data.max_patching_pairs,
            sparse_positions=True,
            output_stem=f"{task}_batch_dnabert_activation_patching",
        )
        outputs[task] = {
            **{key: str(value) for key, value in task_outputs.items()},
            "summary": _batch_patching_summary(task_outputs["table"]),
        }
    return outputs
