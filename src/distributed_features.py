"""Distributed feature search with sparse autoencoders.

The attention-head scans can miss biology that is stored in distributed
residual or MLP features. This module extracts frozen CTCF activations, trains
small sparse autoencoders, and ranks learned dictionary features against
ground-truth JASPAR CTCF motif scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .modeling import LoadedModelBundle, encode_sequences, encoder_layers, forward_hidden_states
from .motif_scoring import score_ctcf_table
from .utils import progress, utc_now_iso, write_json


@dataclass(frozen=True)
class SAETrainingResult:
    """Artifacts produced for one activation family."""

    source: str
    checkpoint_path: Path
    history_path: Path
    alignment_path: Path
    figure_path: Path
    summary: dict[str, Any]


def _as_tensor_output(output: Any):
    """Normalize module outputs that may be tensors or tuples."""

    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Could not locate a tensor in module output of type {type(output)!r}.")


def _find_feed_forward_module(layer: Any) -> Any:
    """Return the closest available feed-forward/MLP module in a DNABERT block."""

    for name in ("mlp", "ffn", "feed_forward", "intermediate"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    attention = getattr(layer, "attention", None)
    output = getattr(attention, "output", None) if attention is not None else None
    if output is not None:
        return output
    return layer


def _mean_pool_any(hidden, attention_mask=None) -> np.ndarray:
    """Mean-pool a single-sequence activation tensor to `[d_model]`."""

    if hidden.ndim == 3:
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled[0].detach().cpu().float().numpy()
    if hidden.ndim == 2:
        return hidden.mean(dim=0).detach().cpu().float().numpy()
    if hidden.ndim == 1:
        return hidden.detach().cpu().float().numpy()
    raise ValueError(f"Unsupported activation rank for pooling: {hidden.ndim}")


def _sequence_motif_scalar(record) -> float:
    """Return one finite motif score per sequence for feature alignment."""

    finite = record.token_scores[np.isfinite(record.token_scores)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite))


def extract_ctcf_residual_and_mlp_activations(
    bundle: LoadedModelBundle,
    config: PipelineConfig = DEFAULT_CONFIG,
    layer_idx: int | None = None,
    max_sequences: int | None = None,
) -> Path:
    """Extract frozen residual and feed-forward activations for CTCF sequences.

    Extraction intentionally runs one sequence at a time so the MLP hook output
    can be pooled correctly even when DNABERT internally unpads tokens.
    """

    import torch

    config.ensure_paths()
    layer_idx = int(config.data.probe_layer if layer_idx is None else layer_idx)
    sequence_limit = max_sequences if max_sequences is not None else config.data.max_feature_search_sequences
    progress(f"Scoring CTCF motif targets for distributed feature search: max_sequences={sequence_limit}")
    motif_records = score_ctcf_table(bundle.tokenizer, config=config, max_sequences=sequence_limit)
    layers = encoder_layers(bundle.hf_model)
    mlp_module = _find_feed_forward_module(layers[layer_idx])
    residual_rows: list[np.ndarray] = []
    mlp_rows: list[np.ndarray] = []
    motif_scores: list[float] = []
    sequence_ids: list[str] = []

    for index, record in enumerate(motif_records):
        captured: dict[str, Any] = {}

        def hook(_module, _inputs, output):
            captured["mlp"] = _as_tensor_output(output).detach()

        handle = mlp_module.register_forward_hook(hook)
        try:
            encoded = encode_sequences(bundle.tokenizer, record.sequence, bundle.device, config.data.token_max_length)
            bundle.hf_model.eval()
            with torch.no_grad():
                hidden_states = forward_hidden_states(bundle.hf_model, encoded)
        finally:
            handle.remove()
        residual_rows.append(_mean_pool_any(hidden_states[layer_idx], encoded.get("attention_mask")))
        if "mlp" not in captured:
            raise RuntimeError("MLP/feed-forward hook did not capture an activation.")
        mlp_rows.append(_mean_pool_any(captured["mlp"], encoded.get("attention_mask")))
        motif_scores.append(_sequence_motif_scalar(record))
        sequence_ids.append(record.sequence_id)
        if index == 0 or index + 1 == len(motif_records) or (index + 1) % 250 == 0:
            progress(f"Distributed feature extraction processed {index + 1}/{len(motif_records)} CTCF sequences")

    output_path = config.paths.distributed_features_dir / "ctcf_layer11_residual_mlp_activations.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        residual=np.asarray(residual_rows, dtype=np.float32),
        mlp=np.asarray(mlp_rows, dtype=np.float32),
        motif_scores=np.asarray(motif_scores, dtype=np.float32),
        sequence_ids=np.asarray(sequence_ids),
        layer=np.asarray([layer_idx], dtype=np.int64),
    )
    write_json(
        config.paths.manifests_dir / "ctcf_distributed_feature_activations_manifest.json",
        {
            "created_at": utc_now_iso(),
            "path": str(output_path),
            "layer": layer_idx,
            "records": len(sequence_ids),
            "sources": ["residual", "mlp"],
        },
    )
    return output_path


class SparseAutoencoder:
    """Small ReLU sparse autoencoder wrapper around a PyTorch module."""

    def __init__(self, input_dim: int, dictionary_size: int):
        import torch

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, dictionary_size),
            torch.nn.ReLU(),
            torch.nn.Linear(dictionary_size, input_dim),
        )

    def to(self, device: str):
        self.model.to(device)
        return self

    def encode(self, x):
        return self.model[1](self.model[0](x))

    def decode(self, z):
        return self.model[2](z)

    def __call__(self, x):
        z = self.encode(x)
        return self.decode(z), z


def _standardize_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standardized activations plus mean/std vectors."""

    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity after mean-centering both vectors."""

    a_centered = a.astype(np.float64) - float(np.mean(a))
    b_centered = b.astype(np.float64) - float(np.mean(b))
    denom = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a_centered, b_centered) / denom)


def _rank_sae_features(encoded_features: np.ndarray, motif_scores: np.ndarray, source: str) -> pd.DataFrame:
    """Rank SAE dictionary features by motif-score alignment."""

    rows = []
    for feature_idx in range(encoded_features.shape[1]):
        similarity = _centered_cosine(encoded_features[:, feature_idx], motif_scores)
        rows.append(
            {
                "source": source,
                "feature": feature_idx,
                "ctcf_motif_cosine": similarity,
                "mean_activation": float(encoded_features[:, feature_idx].mean()),
                "activation_frequency": float((encoded_features[:, feature_idx] > 0).mean()),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("ctcf_motif_cosine", ascending=False, na_position="last")


def _plot_top_features(table: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save a compact bar plot for the top motif-aligned SAE features."""

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = table.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top["feature"].astype(str), top["ctcf_motif_cosine"])
    ax.set_xlabel("centered cosine with CTCF motif score")
    ax.set_ylabel("SAE feature")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def train_sae_and_rank_features(
    activations: np.ndarray,
    motif_scores: np.ndarray,
    source: str,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> SAETrainingResult:
    """Train one SAE and write alignment artifacts for a source activation."""

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    x, mean, std = _standardize_features(np.asarray(activations, dtype=np.float32))
    y = np.asarray(motif_scores, dtype=np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(config.data.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.data.seed))
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(config.data.seed))
    dictionary_size = min(int(config.data.sae_dictionary_size), max(1, x.shape[1] * 4))
    sae = SparseAutoencoder(input_dim=x.shape[1], dictionary_size=dictionary_size).to(device)
    optimizer = torch.optim.AdamW(sae.model.parameters(), lr=float(config.data.sae_learning_rate))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=int(config.data.sae_batch_size),
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )
    history = []
    for epoch in range(int(config.data.sae_epochs)):
        mse_sum = 0.0
        l1_sum = 0.0
        batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, features = sae(batch)
            mse = torch.nn.functional.mse_loss(reconstruction, batch)
            l1 = features.abs().mean()
            loss = mse + float(config.data.sae_l1_coefficient) * l1
            loss.backward()
            optimizer.step()
            mse_sum += float(mse.detach().cpu())
            l1_sum += float(l1.detach().cpu())
            batches += 1
        row = {
            "source": source,
            "epoch": epoch + 1,
            "mse": mse_sum / max(batches, 1),
            "l1": l1_sum / max(batches, 1),
        }
        history.append(row)
        progress(f"SAE {source}: epoch {epoch + 1}/{config.data.sae_epochs}, mse={row['mse']:.5f}, l1={row['l1']:.5f}")

    with torch.no_grad():
        encoded = sae.encode(torch.from_numpy(x).to(device)).detach().cpu().numpy()
    ranking = _rank_sae_features(encoded, y, source=source)
    stem = f"ctcf_{source}_sae"
    checkpoint_path = config.paths.distributed_features_dir / f"{stem}.pt"
    history_path = config.paths.distributed_features_dir / f"{stem}_training_history.csv"
    alignment_path = config.paths.distributed_features_dir / f"{stem}_feature_alignment.csv"
    figure_path = config.paths.figures_dir / f"{stem}_top10_alignment.png"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": sae.model.state_dict(),
            "source": source,
            "input_dim": x.shape[1],
            "dictionary_size": dictionary_size,
            "mean": mean,
            "std": std,
            "config": {
                "epochs": config.data.sae_epochs,
                "l1_coefficient": config.data.sae_l1_coefficient,
                "learning_rate": config.data.sae_learning_rate,
            },
        },
        checkpoint_path,
    )
    pd.DataFrame(history).to_csv(history_path, index=False)
    ranking.to_csv(alignment_path, index=False)
    _plot_top_features(ranking, figure_path, f"Top CTCF-aligned {source} SAE features")
    summary = {
        "source": source,
        "examples": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "dictionary_size": int(dictionary_size),
        "top_feature": int(ranking.iloc[0]["feature"]) if not ranking.empty else None,
        "top_ctcf_motif_cosine": float(ranking.iloc[0]["ctcf_motif_cosine"]) if not ranking.empty else None,
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "alignment": str(alignment_path),
        "figure": str(figure_path),
    }
    return SAETrainingResult(
        source=source,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        alignment_path=alignment_path,
        figure_path=figure_path,
        summary=summary,
    )


def _combined_top_feature_table(results: list[SAETrainingResult], top_n: int = 10) -> pd.DataFrame:
    """Return the globally top motif-aligned SAE features across sources."""

    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    tables = [pd.read_csv(result.alignment_path) for result in results]
    if not tables:
        return pd.DataFrame()
    combined = pd.concat(tables, ignore_index=True)
    return combined.sort_values("ctcf_motif_cosine", ascending=False, na_position="last").head(top_n)


def run_distributed_feature_search(
    bundle: LoadedModelBundle,
    config: PipelineConfig = DEFAULT_CONFIG,
    max_sequences: int | None = None,
) -> dict[str, Any]:
    """Run residual/MLP SAE feature search and write summary artifacts."""

    activation_path = extract_ctcf_residual_and_mlp_activations(
        bundle=bundle,
        config=config,
        max_sequences=max_sequences,
    )
    payload = np.load(activation_path, allow_pickle=True)
    motif_scores = np.asarray(payload["motif_scores"], dtype=np.float32)
    results = [
        train_sae_and_rank_features(payload["residual"], motif_scores, "residual", config=config),
        train_sae_and_rank_features(payload["mlp"], motif_scores, "mlp", config=config),
    ]
    combined = _combined_top_feature_table(results, top_n=10)
    combined_path = config.paths.distributed_features_dir / "ctcf_sae_feature_alignment_top10.csv"
    combined.to_csv(combined_path, index=False)
    summary = {
        "created_at": utc_now_iso(),
        "activation_path": str(activation_path),
        "combined_top10_alignment": str(combined_path),
        "sources": {result.source: result.summary for result in results},
    }
    summary_path = config.paths.distributed_features_dir / "ctcf_distributed_feature_search_summary.json"
    write_json(summary_path, summary)
    return {
        "activation_path": activation_path,
        "summary_path": summary_path,
        "combined_alignment_path": combined_path,
        "sources": {
            result.source: {
                "checkpoint": result.checkpoint_path,
                "history": result.history_path,
                "alignment": result.alignment_path,
                "figure": result.figure_path,
                "summary": result.summary,
            }
            for result in results
        },
    }
