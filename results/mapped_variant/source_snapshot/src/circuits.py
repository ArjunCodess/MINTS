"""QK and OV circuit extraction for DNABERT-style models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

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
    available_layers = encoder_layers(bundle.hf_model)
    layer_indices = tuple(layers) if layers is not None else tuple(config.data.circuit_layers)
    if not layer_indices:
        layer_indices = tuple(range(len(available_layers)))
    invalid_layers = [layer_idx for layer_idx in layer_indices if layer_idx < 0 or layer_idx >= len(available_layers)]
    if invalid_layers:
        raise ValueError(
            f"Requested circuit layers {invalid_layers} outside available encoder layer range "
            f"0..{len(available_layers) - 1}."
        )
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


def _load_probe_direction(task: str, config: PipelineConfig) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Train a residual-stream probe and return its direction in original coordinates.

    The linear probe is fit inside a `StandardScaler -> LogisticRegression`
    pipeline. Scikit-learn stores the logistic coefficient in standardized
    coordinates, so the residual-space direction is `coef / scaler.scale_`.
    That is the vector that should be compared against residual-stream write
    directions such as OV output singular vectors.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .data_ingestion import canonicalize_task_name
    from .probing import _features_for_layer, _load_activation_file

    canonical_task = canonicalize_task_name(task, config.data)
    train_payload = _load_activation_file(config.paths.activations_dir / f"{canonical_task}_train_residual_mean.npz")
    x_train, y_train = _features_for_layer(train_payload, config.data.probe_layer)
    if np.unique(y_train).size < 2:
        raise ValueError(f"Cannot train OV readout probe for {canonical_task}: only one class is present.")

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=config.data.seed,
            solver="lbfgs",
        ),
    )
    classifier.fit(x_train, y_train)
    scaler = classifier.named_steps["standardscaler"]
    logistic = classifier.named_steps["logisticregression"]
    direction = np.asarray(logistic.coef_[0], dtype=np.float64) / np.asarray(scaler.scale_, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        raise ValueError(f"Trained probe direction for {canonical_task} has near-zero norm.")
    direction = direction / norm
    metadata: dict[str, float | int | str] = {
        "task": canonical_task,
        "probe_layer": int(config.data.probe_layer),
        "train_examples": int(len(y_train)),
        "train_positive_rate": float(np.mean(y_train)),
        "direction_norm_before_normalization": float(norm),
    }
    return direction, metadata


def ov_probe_alignment_table(
    ov_matrix: np.ndarray,
    probe_direction: np.ndarray,
    layer: int,
    head: int,
    top_n: int = 25,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compare one OV matrix's singular directions to a probe direction.

    `ov_matrix` is stored for row-vector residual updates, `x @ OV`. Under this
    convention, right singular vectors (`V^T` rows from NumPy SVD) are the
    residual-stream output/write directions. Left singular vectors are retained
    as input/read-direction diagnostics.
    """

    ov = np.asarray(ov_matrix, dtype=np.float64)
    probe = np.asarray(probe_direction, dtype=np.float64).reshape(-1)
    if ov.ndim != 2 or ov.shape[0] != ov.shape[1]:
        raise ValueError(f"OV matrix must be square rank-2; got shape {ov.shape}.")
    if probe.shape[0] != ov.shape[1]:
        raise ValueError(f"Probe direction length {probe.shape[0]} does not match OV width {ov.shape[1]}.")
    probe_norm = np.linalg.norm(probe)
    if probe_norm <= 1e-12:
        raise ValueError("Probe direction has near-zero norm.")
    probe = probe / probe_norm

    u, singular_values, vt = np.linalg.svd(ov, full_matrices=False)
    rows = []
    for idx, singular_value in enumerate(singular_values):
        input_cosine = float(np.dot(u[:, idx], probe))
        output_cosine = float(np.dot(vt[idx, :], probe))
        rows.append(
            {
                "layer": int(layer),
                "head": int(head),
                "singular_index": int(idx),
                "singular_value": float(singular_value),
                "input_left_singular_cosine": input_cosine,
                "input_left_singular_abs_cosine": abs(input_cosine),
                "output_write_singular_cosine": output_cosine,
                "output_write_singular_abs_cosine": abs(output_cosine),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["output_write_singular_abs_cosine", "singular_value"],
        ascending=[False, False],
    )
    if top_n > 0:
        table = table.head(top_n)

    probe_read_direction = ov @ probe
    summary = {
        "layer": int(layer),
        "head": int(head),
        "d_model": int(ov.shape[0]),
        "singular_vectors": int(len(singular_values)),
        "top_output_write_abs_cosine": float(np.max(np.abs(vt @ probe))),
        "top_input_left_abs_cosine": float(np.max(np.abs(u.T @ probe))),
        "spectral_norm": float(singular_values[0]) if singular_values.size else float("nan"),
        "probe_self_gain": float(probe @ ov @ probe),
        "probe_read_direction_norm": float(np.linalg.norm(probe_read_direction)),
    }
    return table.reset_index(drop=True), summary


def analyze_tata_ov_readout(
    config: PipelineConfig = DEFAULT_CONFIG,
    circuit_path: Path | None = None,
    task: str = "promoter_tata",
    layer: int = 2,
    head: int = 7,
    top_n: int = 25,
) -> dict[str, Path | dict[str, float | int | str]]:
    """Analyze whether the TATA-restoring head writes along the TATA probe direction."""

    config.ensure_paths()
    archive_path = circuit_path or (config.paths.circuits_dir / "qk_ov_matrices.npz")
    if not archive_path.exists():
        raise FileNotFoundError(f"Circuit archive not found: {archive_path}")

    progress(f"Analyzing OV readout alignment for {task}: layer={layer}, head={head}")
    with np.load(archive_path) as archive:
        if "ov" not in archive or "layers" not in archive:
            raise ValueError(f"Circuit archive {archive_path} must contain 'ov' and 'layers' arrays.")
        layers = archive["layers"].astype(int).tolist()
        if layer not in layers:
            raise ValueError(f"Layer {layer} is not present in circuit archive layers={layers}.")
        layer_offset = layers.index(layer)
        ov = np.asarray(archive["ov"][layer_offset, head], dtype=np.float64)

    probe_direction, probe_metadata = _load_probe_direction(task, config)
    table, summary = ov_probe_alignment_table(
        ov_matrix=ov,
        probe_direction=probe_direction,
        layer=layer,
        head=head,
        top_n=top_n,
    )
    summary.update(probe_metadata)

    table_path = config.paths.tables_dir / "tata_l2h7_ov_probe_alignment.csv"
    figure_path = config.paths.figures_dir / "tata_l2h7_ov_probe_alignment.png"
    manifest_path = config.paths.manifests_dir / "tata_l2h7_ov_readout_manifest.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)

    import matplotlib.pyplot as plt

    plot_table = table.sort_values("output_write_singular_abs_cosine", ascending=True)
    labels = [str(int(idx)) for idx in plot_table["singular_index"]]
    fig, ax = plt.subplots(figsize=(7.5, max(4.0, len(plot_table) * 0.22)))
    ax.barh(labels, plot_table["output_write_singular_abs_cosine"])
    ax.set_xlabel("abs cosine with TATA probe direction")
    ax.set_ylabel("OV singular-vector index")
    ax.set_title("TATA L2H7 OV write-direction alignment")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    write_json(
        manifest_path,
        {
            "created_at": utc_now_iso(),
            "circuit_archive": str(archive_path),
            "table_path": str(table_path),
            "figure_path": str(figure_path),
            "summary": summary,
            "interpretation_note": (
                "output_write_singular_abs_cosine compares the TATA probe direction "
                "to OV right singular vectors under the row-vector residual update x @ OV."
            ),
        },
    )
    progress(
        "Finished TATA OV readout analysis: "
        f"top_output_abs_cos={summary['top_output_write_abs_cosine']:.4f}, "
        f"probe_self_gain={summary['probe_self_gain']:.4f}"
    )
    return {
        "table": table_path,
        "figure": figure_path,
        "manifest": manifest_path,
        "summary": summary,
    }
