"""QK circuit alignment against ground-truth CTCF motif scores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .enrichment import MotifSupport, _matched_background_indices
from .modeling import LoadedModelBundle, encode_sequences, encoder_layers, infer_attention_geometry
from .motif_scoring import TokenMotifScores, score_ctcf_table, save_token_motif_scores
from .utils import progress, utc_now_iso, write_json


@dataclass(frozen=True)
class AlignmentThresholds:
    """Pre-registered QK-to-motif candidate thresholds."""

    min_r: float = 0.5
    max_p: float = 0.05


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute Pearson r and p-value while ignoring NaNs."""

    from scipy.stats import pearsonr

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return np.nan, np.nan
    if np.nanstd(x[valid]) == 0 or np.nanstd(y[valid]) == 0:
        return np.nan, np.nan
    result = pearsonr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue)


def qk_key_scores(hidden_states: np.ndarray, qk_matrices: np.ndarray, d_head: int | None = None) -> np.ndarray:
    """Compute query-averaged QK key scores for each head and key position.

    Args:
        hidden_states: Layer-input hidden states shaped `[tokens, d_model]`.
        qk_matrices: QK matrices shaped `[head, d_model, d_model]`.
        d_head: Optional head dimension for Transformer attention scaling.

    Returns:
        Array shaped `[head, tokens]`; each value is the mean attention logit
        assigned to that key position across all query positions.
    """

    hidden = np.asarray(hidden_states, dtype=np.float32)
    qk = np.asarray(qk_matrices, dtype=np.float32)
    if hidden.ndim != 2:
        raise ValueError("hidden_states must be shaped [tokens, d_model].")
    if qk.ndim != 3:
        raise ValueError("qk_matrices must be shaped [head, d_model, d_model].")
    if hidden.shape[1] != qk.shape[1] or qk.shape[1] != qk.shape[2]:
        raise ValueError("QK matrices and hidden states have incompatible d_model dimensions.")

    scale = np.sqrt(float(d_head or max(1, hidden.shape[1] // qk.shape[0])))
    # score[h, query, key] = hidden[query] @ qk[h] @ hidden[key]
    scores = np.einsum("qd,hde,ke->hqk", hidden, qk, hidden) / scale
    return scores.mean(axis=1)


def qk_attention_maps(hidden_states: np.ndarray, qk_matrices: np.ndarray, d_head: int | None = None) -> np.ndarray:
    """Reconstruct head-wise attention probabilities from hidden states and QK circuits.

    Returns:
        Array shaped `[head, query_pos, key_pos]`, normalized over `key_pos`.
    """

    hidden = np.asarray(hidden_states, dtype=np.float32)
    qk = np.asarray(qk_matrices, dtype=np.float32)
    if hidden.ndim != 2:
        raise ValueError("hidden_states must be shaped [tokens, d_model].")
    if qk.ndim != 3:
        raise ValueError("qk_matrices must be shaped [head, d_model, d_model].")
    if hidden.shape[1] != qk.shape[1] or qk.shape[1] != qk.shape[2]:
        raise ValueError("QK matrices and hidden states have incompatible d_model dimensions.")

    scale = np.sqrt(float(d_head or max(1, hidden.shape[1] // qk.shape[0])))
    logits = np.einsum("qd,hde,ke->hqk", hidden, qk, hidden) / scale
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    denominator = weights.sum(axis=-1, keepdims=True)
    return weights / np.clip(denominator, a_min=np.finfo(np.float64).tiny, a_max=None)


def qk_key_scores_from_factors(
    hidden_states: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    d_head: int | None = None,
) -> np.ndarray:
    """Compute query-averaged QK scores via low-rank W_Q/W_K factors."""

    hidden = np.asarray(hidden_states, dtype=np.float32)
    w_q = np.asarray(w_q, dtype=np.float32)
    w_k = np.asarray(w_k, dtype=np.float32)
    if hidden.ndim != 2:
        raise ValueError("hidden_states must be shaped [tokens, d_model].")
    if w_q.shape != w_k.shape or w_q.ndim != 3:
        raise ValueError("w_q and w_k must both be shaped [head, d_model, d_head].")
    if hidden.shape[1] != w_q.shape[1]:
        raise ValueError("QK factors and hidden states have incompatible d_model dimensions.")

    scale = np.sqrt(float(d_head or w_q.shape[-1]))
    q_proj = np.einsum("td,hdf->htf", hidden, w_q)
    k_proj = np.einsum("td,hdf->htf", hidden, w_k)
    scores = np.einsum("hqf,hkf->hqk", q_proj, k_proj) / scale
    return scores.mean(axis=1)


def qk_attention_maps_from_factors(
    hidden_states: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    d_head: int | None = None,
) -> np.ndarray:
    """Reconstruct attention probabilities via low-rank W_Q/W_K factors."""

    hidden = np.asarray(hidden_states, dtype=np.float32)
    w_q = np.asarray(w_q, dtype=np.float32)
    w_k = np.asarray(w_k, dtype=np.float32)
    if hidden.ndim != 2:
        raise ValueError("hidden_states must be shaped [tokens, d_model].")
    if w_q.shape != w_k.shape or w_q.ndim != 3:
        raise ValueError("w_q and w_k must both be shaped [head, d_model, d_head].")
    if hidden.shape[1] != w_q.shape[1]:
        raise ValueError("QK factors and hidden states have incompatible d_model dimensions.")

    scale = np.sqrt(float(d_head or w_q.shape[-1]))
    q_proj = np.einsum("td,hdf->htf", hidden, w_q)
    k_proj = np.einsum("td,hdf->htf", hidden, w_k)
    logits = np.einsum("hqf,hkf->hqk", q_proj, k_proj) / scale
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    denominator = weights.sum(axis=-1, keepdims=True)
    return weights / np.clip(denominator, a_min=np.finfo(np.float64).tiny, a_max=None)


def qk_alignment_table(
    qk_by_layer: np.ndarray,
    layer_indices: Iterable[int],
    hidden_inputs_by_layer: dict[int, list[np.ndarray]],
    motif_records: list[TokenMotifScores],
    d_head: int,
    thresholds: AlignmentThresholds = AlignmentThresholds(),
) -> pd.DataFrame:
    """Correlate each layer/head QK score vector with token motif scores."""

    rows: list[dict[str, Any]] = []
    layer_indices = tuple(int(layer) for layer in layer_indices)
    if qk_by_layer.shape[0] != len(layer_indices):
        raise ValueError("qk_by_layer first dimension must match layer_indices.")

    for layer_offset, layer_idx in enumerate(layer_indices):
        if layer_idx not in hidden_inputs_by_layer:
            raise ValueError(f"Missing hidden inputs for layer {layer_idx}.")
        progress(f"Computing QK/motif correlations for layer {layer_idx}")
        qk_layer = qk_by_layer[layer_offset]
        n_heads = qk_layer.shape[0]
        if len(hidden_inputs_by_layer[layer_idx]) != len(motif_records):
            raise ValueError(
                f"Layer {layer_idx} has {len(hidden_inputs_by_layer[layer_idx])} hidden-state records "
                f"but {len(motif_records)} motif records."
            )
        per_head_scores: list[list[np.ndarray]] = [[] for _ in range(n_heads)]
        motif_vectors: list[np.ndarray] = []
        for hidden, motif_record in zip(hidden_inputs_by_layer[layer_idx], motif_records):
            key_scores = qk_key_scores(hidden, qk_layer, d_head=d_head)
            usable = min(key_scores.shape[1], len(motif_record.token_scores))
            motif_vectors.append(np.asarray(motif_record.token_scores[:usable], dtype=np.float64))
            for head_idx in range(n_heads):
                per_head_scores[head_idx].append(key_scores[head_idx, :usable])

        motif_concat = np.concatenate(motif_vectors) if motif_vectors else np.asarray([], dtype=np.float64)
        for head_idx in range(n_heads):
            head_concat = np.concatenate(per_head_scores[head_idx]) if per_head_scores[head_idx] else np.asarray([])
            r_value, p_value = pearson_correlation(motif_concat, head_concat)
            rows.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "pearson_r": r_value,
                    "p_value": p_value,
                    "n_positions": int(np.isfinite(motif_concat).sum()),
                    "passes_qk_alignment": bool(
                        np.isfinite(r_value)
                        and np.isfinite(p_value)
                        and r_value >= thresholds.min_r
                        and p_value < thresholds.max_p
                    ),
                }
            )
    return pd.DataFrame(rows)


def qk_attention_enrichment_table(
    qk_by_layer: np.ndarray,
    layer_indices: Iterable[int],
    hidden_inputs_by_layer: dict[int, list[np.ndarray]],
    supports: list[MotifSupport],
    d_head: int,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Compute matched enrichment from QK-reconstructed attention maps without a full cache."""

    rows: list[dict[str, Any]] = []
    layer_indices = tuple(int(layer) for layer in layer_indices)
    supports_by_sequence: dict[int, list[MotifSupport]] = {}
    for support in supports:
        supports_by_sequence.setdefault(support.sequence_index, []).append(support)

    for layer_offset, layer_idx in enumerate(layer_indices):
        progress(f"Computing QK-reconstructed enrichment for layer {layer_idx}")
        qk_layer = qk_by_layer[layer_offset]
        n_heads = qk_layer.shape[0]
        motif_mass = np.zeros(n_heads, dtype=np.float64)
        background_mass = np.zeros(n_heads, dtype=np.float64)
        support_tokens = np.zeros(n_heads, dtype=np.int64)
        background_tokens = np.zeros(n_heads, dtype=np.int64)

        for sequence_index, hidden in enumerate(hidden_inputs_by_layer[layer_idx]):
            sequence_supports = supports_by_sequence.get(sequence_index, [])
            if not sequence_supports:
                continue
            attention_heads = qk_attention_maps(hidden, qk_layer, d_head=d_head)
            key_pos = attention_heads.shape[-1]
            for support in sequence_supports:
                start = max(0, support.token_start)
                end = min(key_pos, support.token_end)
                if start >= end:
                    continue
                background = _matched_background_indices(key_pos, start, end)
                if len(background) == 0:
                    continue
                motif_mass += attention_heads[:, :, start:end].sum(axis=(1, 2))
                background_mass += attention_heads[:, :, background].sum(axis=(1, 2))
                support_tokens += end - start
                background_tokens += int(len(background))

        for head_idx in range(n_heads):
            a_motif = (
                float(motif_mass[head_idx] / support_tokens[head_idx])
                if support_tokens[head_idx] > 0
                else np.nan
            )
            a_bg = (
                float(background_mass[head_idx] / background_tokens[head_idx])
                if background_tokens[head_idx] > 0
                else np.nan
            )
            rho = a_motif / a_bg if np.isfinite(a_bg) and a_bg > 0 else np.nan
            rows.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "support_tokens": int(support_tokens[head_idx]),
                    "background_tokens": int(background_tokens[head_idx]),
                    "a_motif": a_motif,
                    "a_bg": a_bg,
                    "rho": rho,
                    "passes_attention_enrichment": bool(np.isfinite(rho) and rho >= threshold),
                }
            )
    return pd.DataFrame(rows)


def _alignment_table_from_stats(
    stats_by_layer: dict[int, dict[str, np.ndarray]],
    thresholds: AlignmentThresholds = AlignmentThresholds(),
) -> pd.DataFrame:
    """Convert online Pearson sufficient statistics into a candidate-head table."""

    from scipy.stats import t as student_t

    rows: list[dict[str, Any]] = []
    for layer_idx, stats in stats_by_layer.items():
        n = stats["n"]
        sum_x = stats["sum_x"]
        sum_y = stats["sum_y"]
        sum_x2 = stats["sum_x2"]
        sum_y2 = stats["sum_y2"]
        sum_xy = stats["sum_xy"]
        numerator = n * sum_xy - sum_x * sum_y
        denominator = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
        with np.errstate(divide="ignore", invalid="ignore"):
            r_values = numerator / denominator
        invalid = (n < 3) | ~np.isfinite(r_values)
        r_values[invalid] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = r_values * np.sqrt((n - 2) / np.clip(1.0 - r_values**2, 1e-15, None))
        p_values = np.full_like(r_values, np.nan, dtype=np.float64)
        valid = ~invalid
        p_values[valid] = 2.0 * student_t.sf(np.abs(t_values[valid]), df=n[valid] - 2)

        for head_idx, (r_value, p_value, n_value) in enumerate(zip(r_values, p_values, n)):
            rows.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "pearson_r": float(r_value) if np.isfinite(r_value) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "n_positions": int(n_value),
                    "passes_qk_alignment": bool(
                        np.isfinite(r_value)
                        and np.isfinite(p_value)
                        and r_value >= thresholds.min_r
                        and p_value < thresholds.max_p
                    ),
                }
            )
    return pd.DataFrame(rows)


def _enrichment_table_from_stats(
    stats_by_layer: dict[int, dict[str, np.ndarray]],
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Convert streaming enrichment sufficient statistics into a table."""

    rows: list[dict[str, Any]] = []
    for layer_idx, stats in stats_by_layer.items():
        motif_mass = stats["motif_mass"]
        background_mass = stats["background_mass"]
        support_tokens = stats["support_tokens"]
        background_tokens = stats["background_tokens"]
        for head_idx in range(len(motif_mass)):
            a_motif = (
                float(motif_mass[head_idx] / support_tokens[head_idx])
                if support_tokens[head_idx] > 0
                else np.nan
            )
            a_bg = (
                float(background_mass[head_idx] / background_tokens[head_idx])
                if background_tokens[head_idx] > 0
                else np.nan
            )
            rho = a_motif / a_bg if np.isfinite(a_bg) and a_bg > 0 else np.nan
            rows.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "support_tokens": int(support_tokens[head_idx]),
                    "background_tokens": int(background_tokens[head_idx]),
                    "a_motif": a_motif,
                    "a_bg": a_bg,
                    "rho": rho,
                    "passes_attention_enrichment": bool(np.isfinite(rho) and rho >= threshold),
                }
            )
    return pd.DataFrame(rows)


def save_qk_alignment_outputs(
    table: pd.DataFrame,
    output_stem: str,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> dict[str, Path]:
    """Save QK alignment table and heatmap artifacts."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    config.ensure_paths()
    table_path = config.paths.qk_alignment_dir / f"{output_stem}.csv"
    figure_path = config.paths.figures_dir / f"{output_stem}_pearson_heatmap.png"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)

    pivot = table.pivot(index="layer", columns="head", values="pearson_r")
    fig, ax = plt.subplots(figsize=(max(6.0, pivot.shape[1] * 0.45), max(4.0, pivot.shape[0] * 0.45)))
    sns.heatmap(pivot, ax=ax, cmap="vlag", center=0.0, annot=False, cbar_kws={"label": "Pearson r"})
    ax.set_title("QK-to-CTCF motif alignment")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    manifest_path = config.paths.manifests_dir / f"{output_stem}_qk_alignment_manifest.json"
    write_json(
        manifest_path,
        {
            "created_at": utc_now_iso(),
            "table_path": str(table_path),
            "figure_path": str(figure_path),
            "candidate_heads": table[table["passes_qk_alignment"]][["layer", "head"]].to_dict("records"),
        },
    )
    return {"table": table_path, "figure": figure_path, "manifest": manifest_path}


def save_qk_attention_enrichment_outputs(
    table: pd.DataFrame,
    output_stem: str,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> dict[str, Path]:
    """Save QK-reconstructed matched attention enrichment artifacts."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    config.ensure_paths()
    table_path = config.paths.enrichment_dir / f"{output_stem}.csv"
    figure_path = config.paths.figures_dir / f"{output_stem}_rho_heatmap.png"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)

    pivot = table.pivot(index="layer", columns="head", values="rho")
    fig, ax = plt.subplots(figsize=(max(6.0, pivot.shape[1] * 0.45), max(4.0, pivot.shape[0] * 0.45)))
    sns.heatmap(pivot, ax=ax, cmap="mako", cbar_kws={"label": "rho_h"})
    ax.set_title("QK-reconstructed CTCF attention enrichment")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    # Keep this manifest name deliberately short. Cross-model runs already put
    # artifacts under a model-specific directory, and appending the full
    # output stem here can exceed Windows/Git path limits for long model IDs.
    manifest_path = config.paths.manifests_dir / "qk_enrich_manifest.json"
    write_json(
        manifest_path,
        {
            "created_at": utc_now_iso(),
            "table_path": str(table_path),
            "figure_path": str(figure_path),
            "candidate_heads": table[table["passes_attention_enrichment"]][["layer", "head"]].to_dict("records"),
        },
    )
    return {"table": table_path, "figure": figure_path, "manifest": manifest_path}


def _encode_one(tokenizer: Any, sequence: str, device: str) -> dict[str, Any]:
    return encode_sequences(tokenizer, sequence, device)


def _capture_layer_inputs(
    bundle: LoadedModelBundle,
    sequence: str,
    layer_indices: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Capture layer-input hidden states for one sequence."""

    captured: dict[int, np.ndarray] = {}
    handles = []
    layers = encoder_layers(bundle.hf_model)

    for layer_idx in layer_indices:
        layer = layers[layer_idx]

        def make_hook(idx: int):
            def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                value = inputs[0].detach().cpu().float().numpy()
                if value.ndim == 3 and value.shape[0] == 1:
                    value = value[0]
                captured[idx] = value

            return hook

        handles.append(layer.register_forward_pre_hook(make_hook(layer_idx)))

    try:
        encoded = _encode_one(bundle.tokenizer, sequence, bundle.device)
        bundle.hf_model.eval()
        import torch

        with torch.no_grad():
            try:
                bundle.hf_model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask"),
                    output_all_encoded_layers=False,
                )
            except TypeError:
                bundle.hf_model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask"),
                    output_hidden_states=True,
                    return_dict=True,
                )
    finally:
        for handle in handles:
            handle.remove()
    return captured


def run_ctcf_qk_alignment(
    bundle: LoadedModelBundle,
    config: PipelineConfig = DEFAULT_CONFIG,
    max_sequences: int | None = 128,
    output_stem: str = "ctcf_qk_alignment",
) -> dict[str, Path]:
    """Run the full CTCF motif/QK alignment export for selected sequences."""

    if max_sequences is not None and max_sequences <= 0:
        max_sequences = None
    qk_archive_path = config.paths.circuits_dir / "qk_ov_matrices.npz"
    if not qk_archive_path.exists():
        raise FileNotFoundError(f"QK/OV matrix archive not found: {qk_archive_path}")
    qk_archive = np.load(qk_archive_path)
    qk_by_layer = qk_archive["qk"]
    w_q_by_layer = qk_archive["w_q"] if "w_q" in qk_archive.files else None
    w_k_by_layer = qk_archive["w_k"] if "w_k" in qk_archive.files else None
    layer_indices = tuple(int(layer) for layer in qk_archive["layers"])
    if len(layer_indices) == 0:
        raise ValueError("QK archive does not contain any layer indices.")
    _, d_head, _ = infer_attention_geometry(bundle.hf_model, layer_indices[0])
    use_low_rank = w_q_by_layer is not None and w_k_by_layer is not None
    if use_low_rank:
        progress("Using low-rank W_Q/W_K factors for strict QK proof scoring")
    else:
        progress("Using dense QK matrices for strict QK proof scoring; rerun circuit export for faster W_Q/W_K factors")

    limit_text = "all" if max_sequences is None else str(max_sequences)
    progress(f"Scoring CTCF motif support for up to {limit_text} sequences")
    motif_records = score_ctcf_table(bundle.tokenizer, config=config, max_sequences=max_sequences)
    motif_scores_path = save_token_motif_scores(
        motif_records,
        f"{output_stem}_token_motif_scores.csv",
        config=config,
    )

    n_heads = int(qk_by_layer.shape[1])
    alignment_stats = {
        layer_idx: {
            "n": np.zeros(n_heads, dtype=np.float64),
            "sum_x": np.zeros(n_heads, dtype=np.float64),
            "sum_y": np.zeros(n_heads, dtype=np.float64),
            "sum_x2": np.zeros(n_heads, dtype=np.float64),
            "sum_y2": np.zeros(n_heads, dtype=np.float64),
            "sum_xy": np.zeros(n_heads, dtype=np.float64),
        }
        for layer_idx in layer_indices
    }
    enrichment_stats = {
        layer_idx: {
            "motif_mass": np.zeros(n_heads, dtype=np.float64),
            "background_mass": np.zeros(n_heads, dtype=np.float64),
            "support_tokens": np.zeros(n_heads, dtype=np.int64),
            "background_tokens": np.zeros(n_heads, dtype=np.int64),
        }
        for layer_idx in layer_indices
    }
    any_supports = False
    report_every = 1000
    for idx, record in enumerate(motif_records, start=1):
        captured = _capture_layer_inputs(bundle, record.sequence, layer_indices)
        motif_scores = np.asarray(record.token_scores, dtype=np.float64)
        if record.support_tokens:
            any_supports = True
        for layer_offset, layer_idx in enumerate(layer_indices):
            hidden = captured[layer_idx]
            qk_layer = qk_by_layer[layer_offset]
            if use_low_rank:
                w_q_layer = w_q_by_layer[layer_offset]
                w_k_layer = w_k_by_layer[layer_offset]
                key_scores = qk_key_scores_from_factors(hidden, w_q_layer, w_k_layer, d_head=d_head)
            else:
                key_scores = qk_key_scores(hidden, qk_layer, d_head=d_head)
            usable = min(key_scores.shape[1], len(motif_scores))
            x = motif_scores[:usable]
            stats = alignment_stats[layer_idx]
            for head_idx in range(n_heads):
                y = key_scores[head_idx, :usable].astype(np.float64, copy=False)
                valid = np.isfinite(x) & np.isfinite(y)
                if not np.any(valid):
                    continue
                x_valid = x[valid]
                y_valid = y[valid]
                stats["n"][head_idx] += len(x_valid)
                stats["sum_x"][head_idx] += x_valid.sum()
                stats["sum_y"][head_idx] += y_valid.sum()
                stats["sum_x2"][head_idx] += np.square(x_valid).sum()
                stats["sum_y2"][head_idx] += np.square(y_valid).sum()
                stats["sum_xy"][head_idx] += (x_valid * y_valid).sum()

            if record.support_tokens:
                if use_low_rank:
                    attention_heads = qk_attention_maps_from_factors(hidden, w_q_layer, w_k_layer, d_head=d_head)
                else:
                    attention_heads = qk_attention_maps(hidden, qk_layer, d_head=d_head)
                key_pos = attention_heads.shape[-1]
                enrich = enrichment_stats[layer_idx]
                for token_idx in record.support_tokens:
                    if token_idx >= key_pos:
                        continue
                    background = _matched_background_indices(key_pos, token_idx, token_idx + 1)
                    if len(background) == 0:
                        continue
                    enrich["motif_mass"] += attention_heads[:, :, token_idx : token_idx + 1].sum(axis=(1, 2))
                    enrich["background_mass"] += attention_heads[:, :, background].sum(axis=(1, 2))
                    enrich["support_tokens"] += 1
                    enrich["background_tokens"] += int(len(background))
        if idx == 1 or idx == len(motif_records) or idx % report_every == 0:
            progress(f"Processed strict QK proof inputs for {idx}/{len(motif_records)} CTCF sequences")

    progress("Computing final QK/motif Pearson tables from streaming statistics")
    table = _alignment_table_from_stats(alignment_stats)
    outputs = save_qk_alignment_outputs(table, output_stem, config=config)
    outputs["motif_scores"] = motif_scores_path

    if any_supports:
        progress("Writing matched CTCF attention enrichment from streaming QK-reconstructed statistics")
        enrichment_table = _enrichment_table_from_stats(enrichment_stats)
        enrichment_outputs = save_qk_attention_enrichment_outputs(
            enrichment_table,
            f"{output_stem}_matched_attention_enrichment",
            config=config,
        )
        outputs["enrichment_table"] = enrichment_outputs["table"]
        outputs["enrichment_figure"] = enrichment_outputs["figure"]
        outputs["enrichment_manifest"] = enrichment_outputs["manifest"]
    else:
        progress("No CTCF motif-support tokens passed the threshold; skipping attention enrichment export")
    return outputs
