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
    """Token-level support interval for a motif instance."""

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
