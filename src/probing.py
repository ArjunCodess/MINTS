"""Linear probing over frozen residual stream vectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .data_ingestion import canonicalize_task_name
from .utils import progress, utc_now_iso, write_json


@dataclass(frozen=True)
class ProbeResult:
    """Metrics for one task probe."""

    task: str
    layer: int
    auroc: float
    auroc_ci_low: float
    auroc_ci_high: float
    auprc: float
    auprc_ci_low: float
    auprc_ci_high: float
    accuracy: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    train_examples: int
    test_examples: int
    train_positive_rate: float
    test_positive_rate: float


def _load_activation_file(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Activation file not found: {path}")
    return dict(np.load(path, allow_pickle=True))


def _features_for_layer(payload: dict[str, np.ndarray], layer: int) -> tuple[np.ndarray, np.ndarray]:
    layers = payload["layers"].astype(int).tolist()
    if layer not in layers:
        raise ValueError(f"Layer {layer} is not in cached layers {layers}.")
    layer_offset = layers.index(layer)
    return payload["residual_mean"][:, layer_offset, :], payload["labels"].astype(int)


def _safe_binary_metric(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Return a metric value, or NaN when a bootstrap sample is single-class."""

    if np.unique(y_true).size < 2:
        return float("nan")
    return float(metric_fn(y_true, scores))


def bootstrap_probe_confidence_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    seed: int,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Estimate deterministic percentile confidence intervals for probe metrics.

    AUROC and AUPRC are undefined for bootstrap samples that contain only one
    class. Those samples are ignored for the class-sensitive metrics but kept
    for accuracy, which remains defined for any resample.
    """

    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    if not (len(y_true) == len(probabilities) == len(predictions)):
        raise ValueError("y_true, probabilities, and predictions must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Cannot bootstrap probe metrics for an empty test set.")
    if n_bootstraps <= 0:
        return {
            "auroc": (float("nan"), float("nan")),
            "auprc": (float("nan"), float("nan")),
            "accuracy": (float("nan"), float("nan")),
        }
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in the open interval (0, 1).")

    rng = np.random.default_rng(seed)
    metrics: dict[str, list[float]] = {"auroc": [], "auprc": [], "accuracy": []}
    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sample_y = y_true[indices]
        sample_prob = probabilities[indices]
        sample_pred = predictions[indices]
        metrics["auroc"].append(_safe_binary_metric(roc_auc_score, sample_y, sample_prob))
        metrics["auprc"].append(_safe_binary_metric(average_precision_score, sample_y, sample_prob))
        metrics["accuracy"].append(float(accuracy_score(sample_y, sample_pred)))

    alpha = 1.0 - confidence_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    intervals: dict[str, tuple[float, float]] = {}
    for name, values in metrics.items():
        finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
        if finite.size == 0:
            intervals[name] = (float("nan"), float("nan"))
        else:
            intervals[name] = (
                float(np.percentile(finite, lower_q)),
                float(np.percentile(finite, upper_q)),
            )
    return intervals


def train_logistic_probe_for_task(
    task: str,
    config: PipelineConfig = DEFAULT_CONFIG,
    layer: int | None = None,
) -> ProbeResult:
    """Train and evaluate a logistic regression probe for one task."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    task = canonicalize_task_name(task, config.data)
    probe_layer = config.data.probe_layer if layer is None else layer
    progress(f"Loading residual caches for probe: task={task}, layer={probe_layer}")
    train_payload = _load_activation_file(config.paths.activations_dir / f"{task}_train_residual_mean.npz")
    test_payload = _load_activation_file(config.paths.activations_dir / f"{task}_test_residual_mean.npz")
    x_train, y_train = _features_for_layer(train_payload, probe_layer)
    x_test, y_test = _features_for_layer(test_payload, probe_layer)
    progress(
        f"Training probe for {task}: train={len(y_train)}, test={len(y_test)}, "
        f"train_pos_rate={float(np.mean(y_train)):.3f}, test_pos_rate={float(np.mean(y_test)):.3f}"
    )

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
    probabilities = classifier.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    auroc = float(roc_auc_score(y_test, probabilities))
    auprc = float(average_precision_score(y_test, probabilities))
    accuracy = float(accuracy_score(y_test, predictions))
    intervals = bootstrap_probe_confidence_intervals(
        y_true=y_test,
        probabilities=probabilities,
        predictions=predictions,
        seed=config.data.seed,
        n_bootstraps=config.data.probe_bootstrap_samples,
        confidence_level=config.data.probe_ci_level,
    )
    progress(f"Probe complete for {task}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, accuracy={accuracy:.4f}")

    return ProbeResult(
        task=task,
        layer=probe_layer,
        auroc=auroc,
        auroc_ci_low=intervals["auroc"][0],
        auroc_ci_high=intervals["auroc"][1],
        auprc=auprc,
        auprc_ci_low=intervals["auprc"][0],
        auprc_ci_high=intervals["auprc"][1],
        accuracy=accuracy,
        accuracy_ci_low=intervals["accuracy"][0],
        accuracy_ci_high=intervals["accuracy"][1],
        train_examples=int(len(y_train)),
        test_examples=int(len(y_test)),
        train_positive_rate=float(np.mean(y_train)),
        test_positive_rate=float(np.mean(y_test)),
    )


def run_all_probes(config: PipelineConfig = DEFAULT_CONFIG) -> Path:
    """Run logistic probes for all configured tasks and save a metrics table."""

    config.ensure_paths()
    results = [train_logistic_probe_for_task(task, config=config) for task in config.data.task_names]
    table = pd.DataFrame([result.__dict__ for result in results])
    output_path = config.paths.tables_dir / "linear_probe_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress(f"Writing probe metrics table: {output_path}")
    table.to_csv(output_path, index=False)
    write_json(
        config.paths.manifests_dir / "linear_probe_manifest.json",
        {
            "created_at": utc_now_iso(),
            "path": str(output_path),
            "layer": int(config.data.probe_layer),
            "tasks": [result.task for result in results],
        },
    )
    return output_path
