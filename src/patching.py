"""Activation patching metrics and artifact writers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .utils import utc_now_iso, write_json


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
