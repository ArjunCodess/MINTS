"""Sequence/pair resampling and multiplicity correction, never token resampling."""
import numpy as np


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected finite p-values in [0, 1]")
    order = np.argsort(p)
    adjusted = np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1))
    result = np.empty_like(p)
    result[order] = np.minimum(1, adjusted)
    return result


def bootstrap_mean_interval(values, seed=1729, repetitions=1000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(len(values), size=len(values))].mean() for _ in range(repetitions)]
    return np.quantile(means, [0.025, 0.975]).tolist()


def validate_metric(name, values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if name in {"pearson_r", "spearman_rho"}:
        valid = (finite >= -1) & (finite <= 1)
    elif name in {"auroc", "auprc", "accuracy", "p_value"}:
        valid = (finite >= 0) & (finite <= 1)
    elif name == "attention_enrichment_ratio":
        valid = finite >= 0
    else:
        raise ValueError(f"Unknown metric: {name}")
    if not valid.all():
        raise ValueError(f"Out-of-range {name}")
