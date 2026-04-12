"""Activation caching for nucleotide transformer analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import DEFAULT_CONFIG, PipelineConfig
from .data_ingestion import canonicalize_task_name
from .modeling import LoadedModelBundle
from .utils import utc_now_iso, write_json


@dataclass(frozen=True)
class ActivationExport:
    """Metadata for a compact activation export."""

    task: str
    split: str
    path: Path
    n_examples: int
    layers: tuple[int, ...]
    pooling: str


def mean_pool_hidden(hidden: Any, attention_mask: Any) -> Any:
    """Mean-pool hidden states over non-padding token positions."""

    mask = attention_mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def mean_pool_unpadded_hidden(hidden: Any, attention_mask: Any) -> Any:
    """Mean-pool DNABERT unpadded hidden states back to one vector per sequence."""

    outputs = []
    cursor = 0
    lengths = attention_mask.sum(dim=1).detach().cpu().tolist()
    for length in lengths:
        length = int(length)
        outputs.append(hidden[cursor : cursor + length].mean(dim=0))
        cursor += length
    import torch

    return torch.stack(outputs, dim=0)


def _selected_dataset_rows(dataset: Any, max_examples: int | None, seed: int) -> Any:
    if max_examples is None:
        return dataset
    if "label" not in dataset.column_names:
        return dataset.select(range(min(max_examples, len(dataset))))

    by_label: dict[int, list[int]] = {}
    for idx, label in enumerate(dataset["label"]):
        by_label.setdefault(int(label), []).append(idx)
    if len(by_label) < 2:
        return dataset.select(range(min(max_examples, len(dataset))))

    rng = np.random.default_rng(seed)
    label_values = sorted(by_label)
    base = max_examples // len(label_values)
    remainder = max_examples % len(label_values)
    selected: list[int] = []
    for offset, label in enumerate(label_values):
        indices = np.asarray(by_label[label], dtype=np.int64)
        rng.shuffle(indices)
        quota = min(len(indices), base + (1 if offset < remainder else 0))
        selected.extend(indices[:quota].tolist())
    rng.shuffle(selected)
    return dataset.select(selected)


def _collate_sequences(tokenizer: Any, sequences: list[str], max_length: int | None) -> dict[str, Any]:
    kwargs = {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
    return tokenizer(sequences, **kwargs)


def cache_task_split_residuals(
    bundle: LoadedModelBundle,
    task: str,
    split: str,
    config: PipelineConfig = DEFAULT_CONFIG,
    layers: Iterable[int] | None = None,
    max_examples: int | None = None,
    pooling: str = "mean",
) -> ActivationExport:
    """Cache compact residual vectors for a saved HF downstream task split.

    The export stores pooled layer outputs rather than full sequence-by-layer
    activation tensors. This keeps the output small enough for routine runs
    while preserving the vectors needed for linear probes.
    """

    import torch
    from datasets import load_from_disk

    if pooling != "mean":
        raise ValueError("Only mean pooling is currently implemented.")

    config.ensure_paths()
    task = canonicalize_task_name(task, config.data)
    layer_indices = tuple(layers or config.data.activation_layers)
    dataset_path = config.paths.hf_downstream_dir / task
    dataset_dict = load_from_disk(str(dataset_path))
    dataset = _selected_dataset_rows(dataset_dict[split], max_examples, config.data.seed)

    pooled_batches: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []
    sequences: list[str] = []
    model = bundle.hf_model
    tokenizer = bundle.tokenizer

    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), config.data.batch_size):
            batch = dataset[start : start + config.data.batch_size]
            batch_sequences = list(batch["sequence"])
            encoded = _collate_sequences(tokenizer, batch_sequences, config.data.token_max_length)
            encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_all_encoded_layers=True,
            )
            encoded_layers = outputs[0]
            pooled_layers = []
            for layer_idx in layer_indices:
                hidden = encoded_layers[layer_idx]
                if hidden.ndim == 3:
                    pooled = mean_pool_hidden(hidden, encoded["attention_mask"])
                elif hidden.ndim == 2:
                    pooled = mean_pool_unpadded_hidden(hidden, encoded["attention_mask"])
                else:
                    raise ValueError(f"Unexpected hidden state rank {hidden.ndim} for layer {layer_idx}.")
                pooled_layers.append(pooled.detach().cpu().numpy())
            pooled_batches.append(np.stack(pooled_layers, axis=1))
            labels.extend(int(label) for label in batch["label"])
            names.extend(str(name) for name in batch["name"])
            sequences.extend(batch_sequences)

    pooled = np.concatenate(pooled_batches, axis=0) if pooled_batches else np.empty((0, len(layer_indices), 0))
    output_path = config.paths.activations_dir / f"{task}_{split}_residual_mean.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        residual_mean=pooled.astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        names=np.asarray(names),
        sequences=np.asarray(sequences),
        layers=np.asarray(layer_indices, dtype=np.int64),
    )
    write_json(
        config.paths.manifests_dir / f"{task}_{split}_activations_manifest.json",
        {
            "created_at": utc_now_iso(),
            "task": task,
            "split": split,
            "path": str(output_path),
            "n_examples": int(len(labels)),
            "layers": list(layer_indices),
            "pooling": pooling,
            "shape": list(pooled.shape),
            "attention_patterns_supported": False,
            "attention_patterns_note": (
                "DNABERT-2 is running through the Hugging Face forward-hook backend; "
                "TransformerLens-style per-head attention-pattern tensors are not exposed."
            ),
            "head_outputs_supported": False,
            "head_outputs_note": (
                "The DNABERT-2 implementation uses unpadded attention internals; "
                "head output decomposition is deferred to the circuit matrix export."
            ),
        },
    )
    return ActivationExport(
        task=task,
        split=split,
        path=output_path,
        n_examples=len(labels),
        layers=layer_indices,
        pooling=pooling,
    )


def cache_probe_residuals(
    bundle: LoadedModelBundle,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> list[ActivationExport]:
    """Cache residual vectors for all configured downstream probing tasks."""

    exports: list[ActivationExport] = []
    for task in config.data.task_names:
        exports.append(
            cache_task_split_residuals(
                bundle=bundle,
                task=task,
                split="train",
                config=config,
                max_examples=config.data.max_probe_train,
            )
        )
        exports.append(
            cache_task_split_residuals(
                bundle=bundle,
                task=task,
                split="test",
                config=config,
                max_examples=config.data.max_probe_test,
            )
        )
    return exports
