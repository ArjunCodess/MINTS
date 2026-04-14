"""Attention enrichment utilities for motif-support positions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .utils import utc_now_iso, write_json


@dataclass(frozen=True)
class MotifSupport:
    """Half-open model-token support interval for one motif instance.

    For DNABERT-2, a support interval may span multiple BPE tokens because a
    JASPAR motif is defined in nucleotide coordinates rather than token
    coordinates.
    """

    sequence_index: int
    token_start: int
    token_end: int
    motif_name: str


def attention_enrichment_ratio(attention: np.ndarray, supports: list[MotifSupport]) -> pd.DataFrame:
    """Compute head-wise attention mass enrichment over motif support tokens.

    Args:
        attention: Array shaped `[batch, layer, head, query_pos, key_pos]`.
        supports: Token intervals `[token_start, token_end)` per sequence.
    """

    if attention.ndim != 5:
        raise ValueError("Expected attention with shape [batch, layer, head, query_pos, key_pos].")
    batch, n_layers, n_heads, _query_pos, key_pos = attention.shape
    rows: list[dict[str, float | int | str]] = []

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            observed = 0.0
            expected = 0.0
            support_tokens = 0
            for support in supports:
                if support.sequence_index >= batch:
                    continue
                start = max(0, support.token_start)
                end = min(key_pos, support.token_end)
                if start >= end:
                    continue
                head_attention = attention[support.sequence_index, layer_idx, head_idx]
                observed += float(head_attention[:, start:end].sum())
                expected += float(head_attention.sum() * ((end - start) / key_pos))
                support_tokens += end - start
            rows.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "support_tokens": support_tokens,
                    "attention_mass": observed,
                    "expected_mass": expected,
                    "enrichment_ratio": observed / expected if expected > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _matched_background_indices(key_pos: int, support_start: int, support_end: int) -> np.ndarray:
    """Select a deterministic non-motif background window matching support length."""

    support = set(range(max(0, support_start), min(key_pos, support_end)))
    width = len(support)
    if width == 0:
        return np.asarray([], dtype=np.int64)
    candidates = [idx for idx in range(key_pos) if idx not in support]
    if len(candidates) < width:
        return np.asarray(candidates, dtype=np.int64)
    support_center = (support_start + support_end - 1) / 2.0
    candidates.sort(key=lambda idx: (abs(idx - support_center), idx))
    return np.asarray(candidates[:width], dtype=np.int64)


def matched_attention_enrichment_ratio(
    attention: np.ndarray,
    supports: list[MotifSupport],
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Compute motif-vs-matched-background attention enrichment.

    Args:
        attention: Array shaped `[batch, layer, head, query_pos, key_pos]`.
        supports: Token intervals `[token_start, token_end)` for motif support.
        threshold: Candidate cutoff for `rho_h = a_motif / a_bg`.
    """

    if attention.ndim != 5:
        raise ValueError("Expected attention with shape [batch, layer, head, query_pos, key_pos].")
    batch, n_layers, n_heads, _query_pos, key_pos = attention.shape
    rows: list[dict[str, float | int | bool]] = []

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            motif_mass = 0.0
            background_mass = 0.0
            support_tokens = 0
            background_tokens = 0
            for support in supports:
                if support.sequence_index >= batch:
                    continue
                start = max(0, support.token_start)
                end = min(key_pos, support.token_end)
                if start >= end:
                    continue
                background = _matched_background_indices(key_pos, start, end)
                if len(background) == 0:
                    continue
                head_attention = attention[support.sequence_index, layer_idx, head_idx]
                motif_mass += float(head_attention[:, start:end].sum())
                background_mass += float(head_attention[:, background].sum())
                support_tokens += end - start
                background_tokens += int(len(background))
            a_motif = motif_mass / support_tokens if support_tokens > 0 else np.nan
            a_bg = background_mass / background_tokens if background_tokens > 0 else np.nan
            rho = a_motif / a_bg if np.isfinite(a_bg) and a_bg > 0 else np.nan
            rows.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "support_tokens": support_tokens,
                    "background_tokens": background_tokens,
                    "a_motif": a_motif,
                    "a_bg": a_bg,
                    "rho": rho,
                    "passes_attention_enrichment": bool(np.isfinite(rho) and rho >= threshold),
                }
            )
    return pd.DataFrame(rows)


def matched_attention_enrichment_from_records(
    attention_by_layer: dict[int, list[np.ndarray]],
    supports: list[MotifSupport],
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Compute matched-background enrichment for variable-length attention maps.

    Args:
        attention_by_layer: Maps each layer index to one array per sequence,
            each shaped `[head, query_pos, key_pos]`.
        supports: Token intervals `[token_start, token_end)` keyed by
            `sequence_index`.
        threshold: Candidate cutoff for `rho_h = a_motif / a_bg`.
    """

    rows: list[dict[str, float | int | bool]] = []
    supports_by_sequence: dict[int, list[MotifSupport]] = {}
    for support in supports:
        supports_by_sequence.setdefault(support.sequence_index, []).append(support)

    for layer_idx, sequence_maps in attention_by_layer.items():
        if not sequence_maps:
            continue
        n_heads = int(sequence_maps[0].shape[0])
        for head_idx in range(n_heads):
            motif_mass = 0.0
            background_mass = 0.0
            support_tokens = 0
            background_tokens = 0
            for sequence_index, attention_heads in enumerate(sequence_maps):
                sequence_supports = supports_by_sequence.get(sequence_index, [])
                if not sequence_supports:
                    continue
                if attention_heads.ndim != 3:
                    raise ValueError("Each attention record must be shaped [head, query_pos, key_pos].")
                key_pos = attention_heads.shape[-1]
                head_attention = attention_heads[head_idx]
                for support in sequence_supports:
                    start = max(0, support.token_start)
                    end = min(key_pos, support.token_end)
                    if start >= end:
                        continue
                    background = _matched_background_indices(key_pos, start, end)
                    if len(background) == 0:
                        continue
                    motif_mass += float(head_attention[:, start:end].sum())
                    background_mass += float(head_attention[:, background].sum())
                    support_tokens += end - start
                    background_tokens += int(len(background))

            a_motif = motif_mass / support_tokens if support_tokens > 0 else np.nan
            a_bg = background_mass / background_tokens if background_tokens > 0 else np.nan
            rho = a_motif / a_bg if np.isfinite(a_bg) and a_bg > 0 else np.nan
            rows.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "support_tokens": support_tokens,
                    "background_tokens": background_tokens,
                    "a_motif": a_motif,
                    "a_bg": a_bg,
                    "rho": rho,
                    "passes_attention_enrichment": bool(np.isfinite(rho) and rho >= threshold),
                }
            )
    return pd.DataFrame(rows)


def save_attention_enrichment(
    attention: np.ndarray,
    supports: list[MotifSupport],
    output_name: str,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Path:
    """Save head-wise attention enrichment ratios to `results/enrichment/`."""

    config.ensure_paths()
    table = attention_enrichment_ratio(attention, supports)
    output_path = config.paths.enrichment_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    write_json(
        config.paths.manifests_dir / f"{Path(output_name).stem}_enrichment_manifest.json",
        {
            "created_at": utc_now_iso(),
            "path": str(output_path),
            "n_supports": len(supports),
            "attention_shape": list(attention.shape),
        },
    )
    return output_path


def save_matched_attention_enrichment(
    attention: np.ndarray,
    supports: list[MotifSupport],
    output_stem: str,
    config: PipelineConfig = DEFAULT_CONFIG,
    threshold: float = 2.0,
) -> dict[str, Path]:
    """Save matched-background enrichment table and heatmap."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    config.ensure_paths()
    table = matched_attention_enrichment_ratio(attention, supports, threshold=threshold)
    table_path = config.paths.enrichment_dir / f"{output_stem}.csv"
    figure_path = config.paths.figures_dir / f"{output_stem}_rho_heatmap.png"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)

    pivot = table.pivot(index="layer", columns="head", values="rho")
    fig, ax = plt.subplots(figsize=(max(6.0, pivot.shape[1] * 0.45), max(4.0, pivot.shape[0] * 0.45)))
    sns.heatmap(pivot, ax=ax, cmap="mako", cbar_kws={"label": "rho_h"})
    ax.set_title("Motif attention enrichment")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    manifest_path = config.paths.manifests_dir / f"{output_stem}_matched_enrichment_manifest.json"
    write_json(
        manifest_path,
        {
            "created_at": utc_now_iso(),
            "table_path": str(table_path),
            "figure_path": str(figure_path),
            "threshold": threshold,
            "candidate_heads": table[table["passes_attention_enrichment"]][["layer", "head"]].to_dict("records"),
        },
    )
    return {"table": table_path, "figure": figure_path, "manifest": manifest_path}
