"""Activation patching metrics and artifact writers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .counterfactuals import MutationRecord, generate_counterfactual_sequence, save_counterfactual_pairs
from .data_ingestion import canonicalize_task_name
from .modeling import encode_sequences, encoder_layers, forward_hidden_states
from .utils import progress, utc_now_iso, write_json


def restoration_metric(
    clean_logit: float | np.ndarray,
    corrupted_logit: float | np.ndarray,
    patched_logit: float | np.ndarray,
    eps: float = 1e-8,
) -> float | np.ndarray:
    """Compute normalized causal restoration.

    The metric is `(patched - corrupted) / (clean - corrupted)`. Near-zero
    denominators are returned as NaN because the clean/corrupted pair did not
    create a measurable target effect to restore.
    """

    clean = np.asarray(clean_logit, dtype=np.float64)
    corrupted = np.asarray(corrupted_logit, dtype=np.float64)
    patched = np.asarray(patched_logit, dtype=np.float64)
    denominator = clean - corrupted
    metric = np.full(np.broadcast_shapes(clean.shape, corrupted.shape, patched.shape), np.nan, dtype=np.float64)
    clean_b = np.broadcast_to(clean, metric.shape)
    corrupted_b = np.broadcast_to(corrupted, metric.shape)
    patched_b = np.broadcast_to(patched, metric.shape)
    denominator_b = np.broadcast_to(denominator, metric.shape)
    valid = np.abs(denominator_b) > eps
    metric[valid] = (patched_b[valid] - corrupted_b[valid]) / denominator_b[valid]
    if metric.shape == ():
        return float(metric)
    return metric


def patch_activation_slice(
    clean_activation: np.ndarray,
    corrupted_activation: np.ndarray,
    positions: slice | list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Patch selected sequence positions from a clean activation into a corrupted activation.

    Expected shapes are `[..., position, d_model]`. Passing `positions=None`
    patches all positions while preserving the corrupted array outside the
    selected slice.
    """

    if clean_activation.shape != corrupted_activation.shape:
        raise ValueError(
            "Clean and corrupted activations must have identical shapes; "
            f"got {clean_activation.shape} and {corrupted_activation.shape}."
        )
    patched = np.array(corrupted_activation, copy=True)
    if positions is None:
        patched[...] = clean_activation
    else:
        patched[..., positions, :] = clean_activation[..., positions, :]
    return patched


def patch_head_output_tensor(
    clean_head_out: np.ndarray,
    corrupted_head_out: np.ndarray,
    layer: int,
    head: int,
    positions: slice | list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Patch one `[layer, head]` slice in a head-output tensor.

    Supported shapes are `[layer, head, position, d_head]` and
    `[batch, layer, head, position, d_head]`.
    """

    if clean_head_out.shape != corrupted_head_out.shape:
        raise ValueError("Clean and corrupted head outputs must have identical shapes.")
    if clean_head_out.ndim not in (4, 5):
        raise ValueError("Expected head output tensor with rank 4 or 5.")

    patched = np.array(corrupted_head_out, copy=True)
    if clean_head_out.ndim == 4:
        if positions is None:
            patched[layer, head, :, :] = clean_head_out[layer, head, :, :]
        else:
            patched[layer, head, positions, :] = clean_head_out[layer, head, positions, :]
    else:
        if positions is None:
            patched[:, layer, head, :, :] = clean_head_out[:, layer, head, :, :]
        else:
            patched[:, layer, head, positions, :] = clean_head_out[:, layer, head, positions, :]
    return patched


def layer_head_restoration_matrix(
    clean_scores: np.ndarray,
    corrupted_scores: np.ndarray,
    patched_scores: np.ndarray,
) -> np.ndarray:
    """Compute mean restoration for patched scores shaped `[layer, head, example]`.

    `clean_scores` and `corrupted_scores` may be scalars or per-example vectors.
    """

    patched = np.asarray(patched_scores, dtype=np.float64)
    if patched.ndim == 2:
        return np.asarray(restoration_metric(clean_scores, corrupted_scores, patched), dtype=np.float64)
    if patched.ndim != 3:
        raise ValueError("Expected patched scores shaped [layer, head] or [layer, head, example].")
    restored = restoration_metric(clean_scores[None, None, :], corrupted_scores[None, None, :], patched)
    return np.nanmean(restored, axis=-1)


def save_restoration_matrix(
    restoration: np.ndarray,
    output_stem: str,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> dict[str, Path]:
    """Save restoration values as CSV plus a heatmap image."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    config.ensure_paths()
    matrix = np.asarray(restoration, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Restoration matrix must be rank 2 with shape [layer, head].")

    table_path = config.paths.patching_dir / f"{output_stem}.csv"
    figure_path = config.paths.figures_dir / f"{output_stem}_heatmap.png"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(
        [
            {"layer": layer_idx, "head": head_idx, "restoration": float(matrix[layer_idx, head_idx])}
            for layer_idx in range(matrix.shape[0])
            for head_idx in range(matrix.shape[1])
        ]
    )
    table.to_csv(table_path, index=False)

    width = max(6.0, matrix.shape[1] * 0.45)
    height = max(4.0, matrix.shape[0] * 0.45)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(matrix, ax=ax, cmap="vlag", center=0.0, cbar_kws={"label": "restoration"})
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title("Activation patching restoration")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    manifest_path = config.paths.manifests_dir / f"{output_stem}_patching_manifest.json"
    write_json(
        manifest_path,
        {
            "created_at": utc_now_iso(),
            "table_path": str(table_path),
            "figure_path": str(figure_path),
            "shape": list(matrix.shape),
            "nan_count": int(np.isnan(matrix).sum()),
        },
    )
    return {"table": table_path, "figure": figure_path, "manifest": manifest_path}


def run_transformerlens_head_out_patching(
    hooked_model,
    clean_tokens,
    corrupted_tokens,
    patching_metric_fn: Callable,
):
    """Run TransformerLens head-output patching when the backend supports it."""

    try:
        from transformer_lens import patching as tl_patching
    except ImportError as exc:
        raise ImportError("TransformerLens patching utilities are not installed.") from exc

    patch_fn = getattr(tl_patching, "get_act_patch_attn_head_out_by_pos", None)
    if patch_fn is None:
        raise RuntimeError(
            "The installed TransformerLens version does not expose "
            "`get_act_patch_attn_head_out_by_pos`."
        )
    if not hasattr(hooked_model, "run_with_cache"):
        raise RuntimeError("The current model backend does not expose `run_with_cache`.")

    _, clean_cache = hooked_model.run_with_cache(clean_tokens)
    return patch_fn(
        hooked_model,
        corrupted_tokens,
        clean_cache,
        patching_metric_fn,
    )


def _train_probe_scorer(task: str, config: PipelineConfig):
    """Train the same linear probe family used as the patching target score."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .probing import _features_for_layer, _load_activation_file

    task = canonicalize_task_name(task, config.data)
    train_payload = _load_activation_file(config.paths.activations_dir / f"{task}_train_residual_mean.npz")
    x_train, y_train = _features_for_layer(train_payload, config.data.probe_layer)
    scorer = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=config.data.seed,
            solver="lbfgs",
        ),
    )
    scorer.fit(x_train, y_train)
    return scorer


def _encode_single(tokenizer, sequence: str, device: str) -> dict:
    return encode_sequences(tokenizer, sequence, device)


def _mean_pool_single(hidden, attention_mask):
    """Mean-pool a single-sequence DNABERT hidden state."""

    if hidden.ndim == 3:
        mask = attention_mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
        return ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).detach().cpu().numpy()
    if hidden.ndim == 2:
        return hidden.mean(dim=0, keepdim=True).detach().cpu().numpy()
    raise ValueError(f"Unexpected hidden rank for probe scoring: {hidden.ndim}")


def _score_sequence_with_probe(bundle, sequence: str, scorer, config: PipelineConfig) -> float:
    import torch

    encoded = _encode_single(bundle.tokenizer, sequence, bundle.device)
    bundle.hf_model.eval()
    with torch.no_grad():
        encoded_layers = forward_hidden_states(bundle.hf_model, encoded)
    hidden = encoded_layers[config.data.probe_layer]
    pooled = _mean_pool_single(hidden, encoded["attention_mask"])
    return float(scorer.decision_function(pooled)[0])


def _cache_clean_attention_self_outputs(bundle, clean_sequence: str, layer_indices: tuple[int, ...]) -> dict[int, object]:
    """Cache concatenated per-head outputs from DNABERT attention-self modules."""

    import torch

    cache: dict[int, object] = {}
    handles = []
    layers = encoder_layers(bundle.hf_model)
    for layer_idx in layer_indices:
        module = layers[layer_idx].attention.self

        def make_hook(idx: int):
            def hook(_module, _inputs, output):
                cache[idx] = output.detach()

            return hook

        handles.append(module.register_forward_hook(make_hook(layer_idx)))
    try:
        encoded = _encode_single(bundle.tokenizer, clean_sequence, bundle.device)
        bundle.hf_model.eval()
        with torch.no_grad():
            bundle.hf_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_all_encoded_layers=False,
            )
    finally:
        for handle in handles:
            handle.remove()
    return cache


def _replace_concatenated_head_slice(output, clean_output, head_idx: int, d_head: int):
    """Replace one contiguous head slice in DNABERT's concatenated head output."""

    start = head_idx * d_head
    end = start + d_head
    if clean_output.shape != output.shape:
        raise ValueError(
            "Clean/corrupted attention output shapes differ; "
            f"got clean {tuple(clean_output.shape)} and corrupted {tuple(output.shape)}."
        )
    patched = output.clone()
    patched[:, start:end] = clean_output[:, start:end]
    return patched


def hierarchical_sparse_patch_positions(
    sequence_length: int,
    motif_token_span: tuple[int, int] | None = None,
    max_positions: int | None = None,
) -> list[int]:
    """Choose sparse token positions for long-context denoising patching.

    The selector keeps all motif-support tokens, then adds logarithmically
    spaced anchors expanding left and right from the motif center. This is a
    practical Stream-inspired pruning rule: it avoids dense all-position
    replacement while preserving local motif evidence and multiscale context.
    """

    if sequence_length <= 0:
        return []
    if motif_token_span is None:
        start = max(0, sequence_length // 2)
        end = min(sequence_length, start + 1)
    else:
        start, end = motif_token_span
        start = max(0, min(sequence_length, int(start)))
        end = max(start + 1, min(sequence_length, int(end)))

    positions: set[int] = set(range(start, end))
    center = (start + end - 1) // 2
    stride = 1
    while len(positions) < sequence_length and (center - stride >= 0 or center + stride < sequence_length):
        if center - stride >= 0:
            positions.add(center - stride)
        if center + stride < sequence_length:
            positions.add(center + stride)
        stride *= 2
        if max_positions is not None and len(positions) >= max_positions:
            break

    positions.add(0)
    positions.add(sequence_length - 1)
    ordered = sorted(positions)
    if max_positions is not None and len(ordered) > max_positions:
        must_keep = sorted(set(range(start, end)) | {0, sequence_length - 1})
        remaining = [pos for pos in ordered if pos not in must_keep]
        ordered = sorted((must_keep + remaining)[:max_positions])
    return ordered


def _replace_concatenated_head_slice_at_positions(
    output,
    clean_output,
    head_idx: int,
    d_head: int,
    positions: list[int] | None = None,
):
    """Replace one head slice at selected token positions."""

    start = head_idx * d_head
    end = start + d_head
    if clean_output.shape != output.shape:
        raise ValueError(
            "Clean/corrupted attention output shapes differ; "
            f"got clean {tuple(clean_output.shape)} and corrupted {tuple(output.shape)}."
        )
    patched = output.clone()
    if positions is None:
        patched[:, start:end] = clean_output[:, start:end]
    else:
        valid_positions = [pos for pos in positions if 0 <= pos < output.shape[0]]
        if valid_positions:
            patched[valid_positions, start:end] = clean_output[valid_positions, start:end]
    return patched


def _patched_probe_score(
    bundle,
    corrupted_sequence: str,
    scorer,
    config: PipelineConfig,
    clean_cache: dict[int, object],
    layer_idx: int,
    head_idx: int,
) -> float:
    """Run corrupted input while replacing one layer/head attention-self output."""

    import torch

    module = encoder_layers(bundle.hf_model)[layer_idx].attention.self
    d_head = int(module.attention_head_size)

    def patch_hook(_module, _inputs, output):
        clean_output = clean_cache[layer_idx].to(device=output.device, dtype=output.dtype)
        return _replace_concatenated_head_slice(output, clean_output, head_idx=head_idx, d_head=d_head)

    handle = module.register_forward_hook(patch_hook)
    try:
        encoded = _encode_single(bundle.tokenizer, corrupted_sequence, bundle.device)
        bundle.hf_model.eval()
        with torch.no_grad():
            encoded_layers = forward_hidden_states(bundle.hf_model, encoded)
    finally:
        handle.remove()

    hidden = encoded_layers[config.data.probe_layer]
    pooled = _mean_pool_single(hidden, encoded["attention_mask"])
    return float(scorer.decision_function(pooled)[0])


def _patched_probe_score_with_positions(
    bundle,
    corrupted_sequence: str,
    scorer,
    config: PipelineConfig,
    clean_cache: dict[int, object],
    layer_idx: int,
    head_idx: int,
    positions: list[int] | None,
) -> float:
    """Patch one layer/head at sparse token positions and score the corrupted run."""

    import torch

    module = encoder_layers(bundle.hf_model)[layer_idx].attention.self
    d_head = int(module.attention_head_size)

    def patch_hook(_module, _inputs, output):
        clean_output = clean_cache[layer_idx].to(device=output.device, dtype=output.dtype)
        return _replace_concatenated_head_slice_at_positions(
            output,
            clean_output,
            head_idx=head_idx,
            d_head=d_head,
            positions=positions,
        )

    handle = module.register_forward_hook(patch_hook)
    try:
        encoded = _encode_single(bundle.tokenizer, corrupted_sequence, bundle.device)
        bundle.hf_model.eval()
        with torch.no_grad():
            encoded_layers = forward_hidden_states(bundle.hf_model, encoded)
    finally:
        handle.remove()

    hidden = encoded_layers[config.data.probe_layer]
    pooled = _mean_pool_single(hidden, encoded["attention_mask"])
    return float(scorer.decision_function(pooled)[0])


def run_custom_dnabert_activation_patching(
    bundle,
    clean_sequence: str,
    corrupted_sequence: str,
    task: str,
    config: PipelineConfig = DEFAULT_CONFIG,
    layer_indices: tuple[int, ...] | None = None,
    output_stem: str = "dnabert_custom_activation_patching",
) -> dict[str, Path]:
    """Run custom forward-hook activation patching over DNABERT-2 heads.

    This uses the attention-self module output shaped `[nonpad_tokens,
    n_heads * d_head]`. Each intervention replaces only one head's contiguous
    slice from the clean run into the corrupted run. The scalar target is the
    task probe's decision-function score on the patched final residual vector.
    """

    task = canonicalize_task_name(task, config.data)
    encoder_layer_list = encoder_layers(bundle.hf_model)
    layers = layer_indices or tuple(range(len(encoder_layer_list)))
    n_heads = int(encoder_layer_list[layers[0]].attention.self.num_attention_heads)
    progress(f"Training probe scorer for activation patching task={task}")
    scorer = _train_probe_scorer(task, config)

    clean_encoded = _encode_single(bundle.tokenizer, clean_sequence, bundle.device)
    corrupted_encoded = _encode_single(bundle.tokenizer, corrupted_sequence, bundle.device)
    if clean_encoded["input_ids"].shape != corrupted_encoded["input_ids"].shape:
        raise ValueError(
            "Clean and corrupted sequences must tokenize to the same shape for head-output patching; "
            f"got {tuple(clean_encoded['input_ids'].shape)} and {tuple(corrupted_encoded['input_ids'].shape)}."
        )

    clean_score = _score_sequence_with_probe(bundle, clean_sequence, scorer, config)
    corrupted_score = _score_sequence_with_probe(bundle, corrupted_sequence, scorer, config)
    clean_cache = _cache_clean_attention_self_outputs(bundle, clean_sequence, layers)

    restoration = np.full((len(layers), n_heads), np.nan, dtype=np.float64)
    for layer_offset, layer_idx in enumerate(layers):
        for head_idx in range(n_heads):
            patched_score = _patched_probe_score(
                bundle=bundle,
                corrupted_sequence=corrupted_sequence,
                scorer=scorer,
                config=config,
                clean_cache=clean_cache,
                layer_idx=layer_idx,
                head_idx=head_idx,
            )
            restoration[layer_offset, head_idx] = restoration_metric(clean_score, corrupted_score, patched_score)
        progress(f"Custom patching complete for layer {layer_idx}")

    outputs = save_restoration_matrix(restoration, output_stem, config=config)
    manifest_path = outputs["manifest"]
    write_json(
        manifest_path,
        {
            "created_at": utc_now_iso(),
            "table_path": str(outputs["table"]),
            "figure_path": str(outputs["figure"]),
            "task": task,
            "target": "linear_probe_decision_function",
            "clean_score": clean_score,
            "corrupted_score": corrupted_score,
            "layers": list(layers),
            "n_heads": n_heads,
            "shape": list(restoration.shape),
            "nan_count": int(np.isnan(restoration).sum()),
        },
    )
    return outputs


def _select_patching_pairs_for_task(
    bundle,
    task: str,
    max_pairs: int,
    config: PipelineConfig,
) -> list[MutationRecord]:
    """Select positive examples whose counterfactuals preserve tokenizer shape."""

    from datasets import load_from_disk

    task = canonicalize_task_name(task, config.data)
    dataset = load_from_disk(str(config.paths.hf_downstream_dir / task))
    records: list[MutationRecord] = []
    for split_name in ("test", "train"):
        if split_name not in dataset:
            continue
        for row_idx, row in enumerate(dataset[split_name]):
            if len(records) >= max_pairs:
                return records
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
            clean = _encode_single(bundle.tokenizer, record.clean_sequence, bundle.device)
            corrupted = _encode_single(bundle.tokenizer, record.corrupted_sequence, bundle.device)
            if clean["input_ids"].shape == corrupted["input_ids"].shape:
                records.append(record)
    return records


def run_batch_dnabert_activation_patching(
    bundle,
    task: str,
    config: PipelineConfig = DEFAULT_CONFIG,
    max_pairs: int | None = None,
    layer_indices: tuple[int, ...] | None = None,
    sparse_positions: bool = True,
    output_stem: str | None = None,
) -> dict[str, Path]:
    """Run denoising activation patching over many clean/corrupted pairs.

    This scales the single-sequence custom hook experiment to a task-level
    estimate. Each pair contributes a per-layer/head restoration score and the
    saved table reports the mean restoration across valid pairs.
    """

    task = canonicalize_task_name(task, config.data)
    pair_limit = int(max_pairs or config.data.max_patching_pairs)
    if pair_limit <= 0:
        raise ValueError("max_pairs must be positive.")

    records = _select_patching_pairs_for_task(bundle, task=task, max_pairs=pair_limit, config=config)
    if not records:
        raise ValueError(f"No token-shape-preserving counterfactual pairs found for {task}.")
    pair_path = save_counterfactual_pairs(records, f"{task}_batch_activation_patching_pairs.tsv", config=config)

    scorer = _train_probe_scorer(task, config)
    encoder_layer_list = encoder_layers(bundle.hf_model)
    layers = layer_indices or tuple(range(len(encoder_layer_list)))
    n_heads = int(encoder_layer_list[layers[0]].attention.self.num_attention_heads)
    per_pair_restoration = np.full((len(records), len(layers), n_heads), np.nan, dtype=np.float64)
    denominator_failures = 0

    for pair_idx, record in enumerate(records):
        clean_score = _score_sequence_with_probe(bundle, record.clean_sequence, scorer, config)
        corrupted_score = _score_sequence_with_probe(bundle, record.corrupted_sequence, scorer, config)
        if abs(clean_score - corrupted_score) <= 1e-8:
            denominator_failures += 1
        clean_cache = _cache_clean_attention_self_outputs(bundle, record.clean_sequence, layers)
        clean_encoded = _encode_single(bundle.tokenizer, record.clean_sequence, bundle.device)
        token_count = int(clean_encoded["attention_mask"].sum().detach().cpu().item())
        positions = None
        if sparse_positions:
            try:
                from .counterfactuals import char_span_to_token_span

                motif_span = char_span_to_token_span(
                    record.clean_sequence,
                    record.start,
                    record.end,
                    bundle.tokenizer,
                )
            except Exception:
                motif_span = None
            positions = hierarchical_sparse_patch_positions(
                sequence_length=token_count,
                motif_token_span=motif_span,
                max_positions=max(8, int(np.ceil(np.log2(max(token_count, 2))) * 4)),
            )

        for layer_offset, layer_idx in enumerate(layers):
            for head_idx in range(n_heads):
                patched_score = _patched_probe_score_with_positions(
                    bundle=bundle,
                    corrupted_sequence=record.corrupted_sequence,
                    scorer=scorer,
                    config=config,
                    clean_cache=clean_cache,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    positions=positions,
                )
                per_pair_restoration[pair_idx, layer_offset, head_idx] = restoration_metric(
                    clean_score,
                    corrupted_score,
                    patched_score,
                )
        if pair_idx == 0 or pair_idx + 1 == len(records) or (pair_idx + 1) % 25 == 0:
            progress(f"{task}: batch patching processed {pair_idx + 1}/{len(records)} pairs")

    restoration = np.nanmean(per_pair_restoration, axis=0)
    stem = output_stem or f"{task}_batch_dnabert_activation_patching"
    outputs = save_restoration_matrix(restoration, stem, config=config)
    write_json(
        outputs["manifest"],
        {
            "created_at": utc_now_iso(),
            "task": task,
            "target": "linear_probe_decision_function",
            "pair_table": str(pair_path),
            "pairs": len(records),
            "denominator_failures": int(denominator_failures),
            "sparse_positions": bool(sparse_positions),
            "layers": list(layers),
            "n_heads": n_heads,
            "table_path": str(outputs["table"]),
            "figure_path": str(outputs["figure"]),
            "shape": list(restoration.shape),
            "nan_count": int(np.isnan(restoration).sum()),
        },
    )
    rows = pd.read_csv(outputs["table"])
    rows["task"] = task
    rows["pairs"] = len(records)
    rows["denominator_failures"] = int(denominator_failures)
    rows.to_csv(outputs["table"], index=False)
    outputs["pairs"] = pair_path
    return outputs
