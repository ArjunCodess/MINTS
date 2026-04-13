"""QK and OV circuit extraction for DNABERT-style models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import DEFAULT_CONFIG, PipelineConfig
from .modeling import LoadedModelBundle, encoder_layers
from .utils import progress, utc_now_iso, write_json


@dataclass(frozen=True)
class CircuitExport:
    """Metadata for exported QK/OV circuit matrices."""

    path: Path
    layers: tuple[int, ...]
    n_heads: int
    d_model: int
    d_head: int


def _get_layer(bundle: LoadedModelBundle, layer_idx: int):
    return encoder_layers(bundle.hf_model)[layer_idx]


def _split_qkv(layer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return W_Q, W_K, W_V as [head, d_model, d_head] arrays."""

    attention = getattr(layer, "attention", None)
    self_attn = getattr(attention, "self", attention)
    if hasattr(self_attn, "Wqkv"):
        weight = self_attn.Wqkv.weight.detach().cpu().float().numpy()
        hidden3, d_model = weight.shape
        hidden = hidden3 // 3
        n_heads = int(self_attn.num_attention_heads)
        d_head = hidden // n_heads
        q, k, v = np.split(weight, 3, axis=0)
        q = q.reshape(n_heads, d_head, d_model).transpose(0, 2, 1)
        k = k.reshape(n_heads, d_head, d_model).transpose(0, 2, 1)
        v = v.reshape(n_heads, d_head, d_model).transpose(0, 2, 1)
        return q, k, v

    q_proj = getattr(self_attn, "query", None) or getattr(self_attn, "q_proj", None)
    k_proj = getattr(self_attn, "key", None) or getattr(self_attn, "k_proj", None)
    v_proj = getattr(self_attn, "value", None) or getattr(self_attn, "v_proj", None)
    if q_proj is None or k_proj is None or v_proj is None:
        raise AttributeError("Could not locate separate Q/K/V projections for this attention layer.")

    q_weight = q_proj.weight.detach().cpu().float().numpy()
    k_weight = k_proj.weight.detach().cpu().float().numpy()
    v_weight = v_proj.weight.detach().cpu().float().numpy()
    out_dim, d_model = q_weight.shape
    n_heads = getattr(self_attn, "num_attention_heads", None) or getattr(self_attn, "num_heads", None)
    if n_heads is None:
        raise AttributeError("Could not infer number of attention heads for this attention layer.")
    n_heads = int(n_heads)
    d_head = out_dim // n_heads

    def split_heads(weight: np.ndarray) -> np.ndarray:
        return weight.reshape(n_heads, d_head, d_model).transpose(0, 2, 1)

    return split_heads(q_weight), split_heads(k_weight), split_heads(v_weight)


def _split_o(layer) -> np.ndarray:
    """Return W_O as [head, d_head, d_model]."""

    attention = getattr(layer, "attention", None)
    self_attn = getattr(attention, "self", attention)
    output = getattr(attention, "output", None)
    out_proj = getattr(output, "dense", None) if output is not None else None
    if out_proj is None:
        out_proj = getattr(self_attn, "out_proj", None) or getattr(attention, "out_proj", None)
    if out_proj is None:
        raise AttributeError("Could not locate attention output projection for this layer.")
    weight = out_proj.weight.detach().cpu().float().numpy()
    d_model, hidden = weight.shape
    n_heads = getattr(self_attn, "num_attention_heads", None) or getattr(self_attn, "num_heads", None)
    if n_heads is None:
        raise AttributeError("Could not infer number of attention heads for this attention layer.")
    n_heads = int(n_heads)
    d_head = hidden // n_heads
    return weight.reshape(d_model, n_heads, d_head).transpose(1, 2, 0)


def extract_qk_ov_matrices(
    bundle: LoadedModelBundle,
    config: PipelineConfig = DEFAULT_CONFIG,
    layers: Iterable[int] | None = None,
    dtype: np.dtype = np.float32,
) -> CircuitExport:
    """Extract exact QK and OV circuit matrices for selected layers."""

    config.ensure_paths()
    layer_indices = tuple(layers or config.data.circuit_layers)
    qk_by_layer = []
    ov_by_layer = []
    w_q_by_layer = []
    w_k_by_layer = []
    n_heads = d_model = d_head = 0

    for layer_idx in layer_indices:
        progress(f"Extracting QK/OV matrices for layer {layer_idx}")
        layer = _get_layer(bundle, layer_idx)
        w_q, w_k, w_v = _split_qkv(layer)
        w_o = _split_o(layer)
        qk = np.einsum("hmd,hnd->hmn", w_q, w_k).astype(dtype)
        ov = np.einsum("hmd,hdn->hmn", w_v, w_o).astype(dtype)
        qk_by_layer.append(qk)
        ov_by_layer.append(ov)
        w_q_by_layer.append(w_q.astype(dtype))
        w_k_by_layer.append(w_k.astype(dtype))
        n_heads, d_model, d_head = w_q.shape

    qk_stack = np.stack(qk_by_layer, axis=0)
    ov_stack = np.stack(ov_by_layer, axis=0)
    w_q_stack = np.stack(w_q_by_layer, axis=0)
    w_k_stack = np.stack(w_k_by_layer, axis=0)
    output_path = config.paths.circuits_dir / "qk_ov_matrices.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress(f"Writing QK/OV matrix archive: {output_path}")
    np.savez_compressed(
        output_path,
        qk=qk_stack,
        ov=ov_stack,
        w_q=w_q_stack,
        w_k=w_k_stack,
        layers=np.asarray(layer_indices, dtype=np.int64),
    )
    write_json(
        config.paths.manifests_dir / "circuits_manifest.json",
        {
            "created_at": utc_now_iso(),
            "path": str(output_path),
            "layers": list(layer_indices),
            "qk_shape": list(qk_stack.shape),
            "ov_shape": list(ov_stack.shape),
            "w_q_shape": list(w_q_stack.shape),
            "w_k_shape": list(w_k_stack.shape),
            "n_heads": int(n_heads),
            "d_model": int(d_model),
            "d_head": int(d_head),
        },
    )
    progress(
        "Finished QK/OV export: "
        f"layers={list(layer_indices)}, heads={n_heads}, d_model={d_model}, d_head={d_head}"
    )
    return CircuitExport(output_path, layer_indices, int(n_heads), int(d_model), int(d_head))
