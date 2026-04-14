"""Activation patching metrics and artifact writers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .counterfactuals import MutationRecord, generate_counterfactual_sequence, save_counterfactual_pairs
from .data_ingestion import canonicalize_task_name
from .modeling import encode_sequences, encoder_layers, forward_hidden_states
from .utils import progress, utc_now_iso, write_json


@dataclass(frozen=True)
class StreamSparseMask:
    """Hierarchical sparse attention-mask estimate for long-context tracing.

    The mask is block-level rather than token-level. Rows are query blocks and
    columns are key blocks at the final refinement resolution. A `True` entry
    means that the key block survived hierarchical top-k pruning for that query
    block.
    """

    mask: np.ndarray
    query_blocks: tuple[tuple[int, int], ...]
    key_blocks: tuple[tuple[int, int], ...]
    selected_key_blocks_by_query: tuple[tuple[int, ...], ...]
    metadata: dict[str, int | float | str | None]


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


def _partition_blocks(length: int, block_size: int) -> tuple[tuple[int, int], ...]:
    """Partition `[0, length)` into contiguous half-open blocks."""

    if length <= 0:
        return ()
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    return tuple((start, min(length, start + block_size)) for start in range(0, length, block_size))


def _overlap_length(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Return token overlap length between two half-open intervals."""

    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _score_stream_key_block(
    key_block: tuple[int, int],
    query_block: tuple[int, int],
    key_scores: np.ndarray,
    motif_token_span: tuple[int, int] | None,
    motif_bonus: float,
    locality_weight: float,
) -> float:
    """Score one candidate key block for a query block during refinement."""

    start, end = key_block
    if start >= end:
        return float("-inf")
    block_scores = key_scores[start:end]
    finite = block_scores[np.isfinite(block_scores)]
    score = float(np.max(finite)) if finite.size else 0.0
    if motif_token_span is not None and _overlap_length(key_block, motif_token_span) > 0:
        score += float(motif_bonus)

    query_center = 0.5 * (query_block[0] + query_block[1] - 1)
    key_center = 0.5 * (start + end - 1)
    score += float(locality_weight) / (1.0 + abs(query_center - key_center))
    return score


def stream_sparse_attention_mask(
    sequence_length: int,
    *,
    key_scores: np.ndarray | None = None,
    motif_token_span: tuple[int, int] | None = None,
    query_block_size: int = 64,
    min_key_block_size: int = 64,
    top_k: int = 4,
    refinement_rounds: int | None = None,
    motif_bonus: float = 1.0,
    locality_weight: float = 0.05,
) -> StreamSparseMask:
    """Estimate a sparse attention mask with STREAM-style block refinement.

    This is a practical hierarchical pruning routine inspired by STREAM-style
    sparse tracing. For each query block, the algorithm starts from the full
    key range, recursively bisects surviving key blocks, scores the child
    blocks, and keeps only the top-k blocks per query at each refinement level.
    With fixed `top_k`, each query block scores `O(top_k log T)` key blocks
    instead of all `T` keys.

    `key_scores` can encode prior importance such as motif scores or cheap
    proxy salience. When omitted, the refinement falls back to motif overlap
    plus local context, which is still useful for long-context denoising
    patching where the biological perturbation span is known.
    """

    if sequence_length <= 0:
        empty = np.zeros((0, 0), dtype=bool)
        return StreamSparseMask(
            mask=empty,
            query_blocks=(),
            key_blocks=(),
            selected_key_blocks_by_query=(),
            metadata={
                "sequence_length": int(sequence_length),
                "query_block_size": int(query_block_size),
                "min_key_block_size": int(min_key_block_size),
                "top_k": int(top_k),
                "refinement_rounds": 0,
                "algorithm": "stream_hierarchical_block_refinement",
            },
        )
    if query_block_size <= 0 or min_key_block_size <= 0:
        raise ValueError("query_block_size and min_key_block_size must be positive.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if key_scores is None:
        score_vector = np.zeros(sequence_length, dtype=np.float64)
    else:
        score_vector = np.asarray(key_scores, dtype=np.float64).reshape(-1)
        if score_vector.shape[0] != sequence_length:
            raise ValueError(
                f"key_scores length {score_vector.shape[0]} does not match sequence_length {sequence_length}."
            )
    if motif_token_span is not None:
        start, end = motif_token_span
        start = max(0, min(sequence_length, int(start)))
        end = max(start + 1, min(sequence_length, int(end)))
        motif_span = (start, end)
    else:
        motif_span = None

    query_blocks = _partition_blocks(sequence_length, query_block_size)
    key_blocks = _partition_blocks(sequence_length, min_key_block_size)
    if refinement_rounds is None:
        refinement_rounds = int(np.ceil(np.log2(max(1, sequence_length / min_key_block_size))))
    refinement_rounds = max(1, int(refinement_rounds))

    mask = np.zeros((len(query_blocks), len(key_blocks)), dtype=bool)
    selected_by_query: list[tuple[int, ...]] = []
    for query_idx, query_block in enumerate(query_blocks):
        candidates: list[tuple[int, int]] = [(0, sequence_length)]
        for _ in range(refinement_rounds):
            children: list[tuple[int, int]] = []
            for start, end in candidates:
                if end - start <= min_key_block_size:
                    children.append((start, end))
                    continue
                midpoint = start + ((end - start) // 2)
                midpoint = max(start + 1, min(end - 1, midpoint))
                children.append((start, midpoint))
                children.append((midpoint, end))
            ranked = sorted(
                children,
                key=lambda block: (
                    _score_stream_key_block(
                        block,
                        query_block,
                        score_vector,
                        motif_span,
                        motif_bonus,
                        locality_weight,
                    ),
                    -(block[1] - block[0]),
                    -block[0],
                ),
                reverse=True,
            )
            candidates = ranked[: min(top_k, len(ranked))]
            if all(end - start <= min_key_block_size for start, end in candidates):
                break

        selected_indices: set[int] = set()
        for candidate in candidates:
            first_block = max(0, candidate[0] // min_key_block_size)
            last_block = min(len(key_blocks), int(np.ceil(candidate[1] / min_key_block_size)))
            selected_indices.update(range(first_block, last_block))
        if motif_span is not None:
            first_block = max(0, motif_span[0] // min_key_block_size)
            last_block = min(len(key_blocks), int(np.ceil(motif_span[1] / min_key_block_size)))
            selected_indices.update(range(first_block, last_block))
        selected_tuple = tuple(sorted(selected_indices))
        selected_by_query.append(selected_tuple)
        if selected_tuple:
            mask[query_idx, list(selected_tuple)] = True

    metadata = {
        "sequence_length": int(sequence_length),
        "query_block_size": int(query_block_size),
        "min_key_block_size": int(min_key_block_size),
        "top_k": int(top_k),
        "refinement_rounds": int(refinement_rounds),
        "query_blocks": int(len(query_blocks)),
        "key_blocks": int(len(key_blocks)),
        "selected_block_entries": int(mask.sum()),
        "density": float(mask.mean()) if mask.size else 0.0,
        "algorithm": "stream_hierarchical_block_refinement",
    }
    return StreamSparseMask(
        mask=mask,
        query_blocks=query_blocks,
        key_blocks=key_blocks,
        selected_key_blocks_by_query=tuple(selected_by_query),
        metadata=metadata,
    )


def stream_sparse_patch_positions(
    sequence_length: int,
    motif_token_span: tuple[int, int] | None = None,
    max_positions: int | None = None,
    top_k: int = 4,
    min_key_block_size: int | None = None,
) -> list[int]:
    """Convert a STREAM-style block mask into sparse token patch positions.

    Head-output patching replaces query-position activations, while STREAM
    produces query-key block structure. For intervention efficiency we retain
    token positions from selected query blocks near the perturbation and from
    their surviving key blocks. This keeps the known motif span, sequence
    endpoints, and the highest-priority blocks under a `max_positions` cap.
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

    if max_positions is not None and max_positions <= 0:
        raise ValueError("max_positions must be positive when provided.")
    if min_key_block_size is None:
        if max_positions is None:
            min_key_block_size = max(1, int(np.ceil(np.sqrt(sequence_length))))
        else:
            min_key_block_size = max(1, int(np.ceil(sequence_length / max(max_positions, 1))))
    query_block_size = max(1, int(min_key_block_size))
    sparse_mask = stream_sparse_attention_mask(
        sequence_length=sequence_length,
        motif_token_span=(start, end),
        query_block_size=query_block_size,
        min_key_block_size=int(min_key_block_size),
        top_k=top_k,
    )

    motif_query_rows = [
        idx for idx, block in enumerate(sparse_mask.query_blocks) if _overlap_length(block, (start, end)) > 0
    ]
    if not motif_query_rows and sparse_mask.query_blocks:
        center = (start + end - 1) // 2
        motif_query_rows = [
            min(
                range(len(sparse_mask.query_blocks)),
                key=lambda idx: abs(0.5 * (sparse_mask.query_blocks[idx][0] + sparse_mask.query_blocks[idx][1] - 1) - center),
            )
        ]

    positions: set[int] = set(range(start, end)) | {0, sequence_length - 1}
    for query_idx in motif_query_rows:
        query_block = sparse_mask.query_blocks[query_idx]
        positions.update(range(query_block[0], query_block[1]))
        for key_block_idx in sparse_mask.selected_key_blocks_by_query[query_idx]:
            key_block = sparse_mask.key_blocks[key_block_idx]
            positions.update(range(key_block[0], key_block[1]))

    ordered = sorted(pos for pos in positions if 0 <= pos < sequence_length)
    if max_positions is not None and len(ordered) > max_positions:
        must_keep = sorted(set(range(start, end)) | {0, sequence_length - 1})
        center = (start + end - 1) // 2
        remaining = sorted(
            [pos for pos in ordered if pos not in must_keep],
            key=lambda pos: (abs(pos - center), pos),
        )
        budget = max(0, max_positions - len(must_keep))
        ordered = sorted(must_keep + remaining[:budget])
    return ordered


def hierarchical_sparse_patch_positions(
    sequence_length: int,
    motif_token_span: tuple[int, int] | None = None,
    max_positions: int | None = None,
) -> list[int]:
    """Backward-compatible wrapper around the STREAM sparse selector."""

    return stream_sparse_patch_positions(
        sequence_length=sequence_length,
        motif_token_span=motif_token_span,
        max_positions=max_positions,
    )


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
    sparse_position_counts: list[int] = []

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
            positions = stream_sparse_patch_positions(
                sequence_length=token_count,
                motif_token_span=motif_span,
                max_positions=max(8, int(np.ceil(np.log2(max(token_count, 2))) * 4)),
                top_k=4,
            )
            sparse_position_counts.append(len(positions))

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
            "sparse_mask_algorithm": (
                "stream_hierarchical_block_refinement" if sparse_positions else None
            ),
            "sparse_position_count_min": int(min(sparse_position_counts)) if sparse_position_counts else None,
            "sparse_position_count_max": int(max(sparse_position_counts)) if sparse_position_counts else None,
            "sparse_position_count_mean": (
                float(np.mean(sparse_position_counts)) if sparse_position_counts else None
            ),
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
