"""Linear probing over frozen residual stream vectors."""

from __future__ import annotations

import re
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


@dataclass(frozen=True)
class ProbeControlResult:
    """Metrics for one probe interpretability control."""

    task: str
    control: str
    layer: int
    auroc: float
    auprc: float
    accuracy: float
    train_examples: int
    test_examples: int
    train_positive_rate: float
    test_positive_rate: float
    run: int | None = None
    notes: str = ""


_COORD_RE = re.compile(
    r"(?P<chrom>chr(?:[0-9]{1,2}|x|y|m|mt)):(?P<start>[0-9]+)-(?P<end>[0-9]+)",
    flags=re.IGNORECASE,
)


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


def _sequence_gc_fraction(sequence: str) -> float:
    sequence = str(sequence).upper()
    valid = sum(base in {"A", "C", "G", "T"} for base in sequence)
    if valid == 0:
        return float("nan")
    return float((sequence.count("G") + sequence.count("C")) / valid)


def _gc_content_features(sequences: np.ndarray) -> np.ndarray:
    """Return sequence-composition features that discard order and residual state."""

    rows: list[list[float]] = []
    for sequence in sequences:
        text = str(sequence).upper()
        valid = max(1, sum(base in {"A", "C", "G", "T"} for base in text))
        g_count = text.count("G")
        c_count = text.count("C")
        gc_fraction = (g_count + c_count) / valid
        gc_skew = (g_count - c_count) / max(1, g_count + c_count)
        rows.append([float(gc_fraction), float(gc_skew), float(len(text))])
    return np.asarray(rows, dtype=np.float64)


def _chromosome_number(chromosome: str) -> float:
    value = chromosome.lower().removeprefix("chr")
    if value == "x":
        return 23.0
    if value == "y":
        return 24.0
    if value in {"m", "mt"}:
        return 25.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _position_features(names: np.ndarray) -> tuple[np.ndarray, float]:
    """Extract coordinate-only features when dataset names include genomic loci."""

    rows: list[list[float]] = []
    parsed = 0
    for name in names:
        match = _COORD_RE.search(str(name))
        if match is None:
            rows.append([float("nan")] * 5)
            continue
        start = float(match.group("start"))
        end = float(match.group("end"))
        midpoint = (start + end) / 2.0
        rows.append(
            [
                _chromosome_number(match.group("chrom")),
                start,
                end,
                midpoint,
                max(0.0, end - start),
            ]
        )
        parsed += 1
    features = np.asarray(rows, dtype=np.float64)
    coverage = parsed / max(1, len(names))
    return features, float(coverage)


def _classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
    }


def _fit_evaluate_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, float] | None:
    """Fit a balanced logistic classifier and return metrics, or None if undefined."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)
    if x_train.size == 0 or x_test.size == 0:
        return None
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        return None
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
        ),
    )
    classifier.fit(x_train, y_train)
    probabilities = classifier.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return _classification_metrics(y_test, probabilities, predictions)


def _nan_control_result(
    task: str,
    control: str,
    layer: int,
    y_train: np.ndarray,
    y_test: np.ndarray,
    notes: str,
    run: int | None = None,
) -> ProbeControlResult:
    return ProbeControlResult(
        task=task,
        control=control,
        layer=layer,
        auroc=float("nan"),
        auprc=float("nan"),
        accuracy=float("nan"),
        train_examples=int(len(y_train)),
        test_examples=int(len(y_test)),
        train_positive_rate=float(np.mean(y_train)) if len(y_train) else float("nan"),
        test_positive_rate=float(np.mean(y_test)) if len(y_test) else float("nan"),
        run=run,
        notes=notes,
    )


def _control_result_from_metrics(
    task: str,
    control: str,
    layer: int,
    metrics: dict[str, float] | None,
    y_train: np.ndarray,
    y_test: np.ndarray,
    notes: str = "",
    run: int | None = None,
) -> ProbeControlResult:
    if metrics is None:
        return _nan_control_result(
            task=task,
            control=control,
            layer=layer,
            y_train=y_train,
            y_test=y_test,
            notes=notes or "skipped because train or test labels were single-class",
            run=run,
        )
    return ProbeControlResult(
        task=task,
        control=control,
        layer=layer,
        auroc=metrics["auroc"],
        auprc=metrics["auprc"],
        accuracy=metrics["accuracy"],
        train_examples=int(len(y_train)),
        test_examples=int(len(y_test)),
        train_positive_rate=float(np.mean(y_train)),
        test_positive_rate=float(np.mean(y_test)),
        run=run,
        notes=notes,
    )


def _matched_gc_indices(
    sequences: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Return a balanced test subset with one nearest-GC negative per positive."""

    labels = np.asarray(labels, dtype=int)
    gc_values = np.asarray([_sequence_gc_fraction(sequence) for sequence in sequences], dtype=np.float64)
    finite = np.isfinite(gc_values)
    positive = np.flatnonzero((labels == 1) & finite)
    negative = np.flatnonzero((labels == 0) & finite)
    if positive.size == 0 or negative.size == 0:
        return np.asarray([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    rng.shuffle(positive)
    available = negative.tolist()
    selected: list[int] = []
    for pos_idx in positive.tolist():
        if not available:
            break
        neg_gc = gc_values[np.asarray(available, dtype=np.int64)]
        best_offset = int(np.argmin(np.abs(neg_gc - gc_values[pos_idx])))
        neg_idx = available.pop(best_offset)
        selected.extend([pos_idx, int(neg_idx)])
    selected_array = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_array)
    return selected_array


def _gc_shift_indices(
    train_sequences: np.ndarray,
    test_sequences: np.ndarray,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    train_gc = np.asarray([_sequence_gc_fraction(sequence) for sequence in train_sequences], dtype=np.float64)
    test_gc = np.asarray([_sequence_gc_fraction(sequence) for sequence in test_sequences], dtype=np.float64)
    finite_train = np.isfinite(train_gc)
    finite_test = np.isfinite(test_gc)
    low_threshold = float(np.nanquantile(train_gc[finite_train], 0.40))
    high_threshold = float(np.nanquantile(train_gc[finite_train], 0.60))
    if direction == "low_to_high":
        return (
            np.flatnonzero(finite_train & (train_gc <= low_threshold)),
            np.flatnonzero(finite_test & (test_gc >= high_threshold)),
        )
    if direction == "high_to_low":
        return (
            np.flatnonzero(finite_train & (train_gc >= high_threshold)),
            np.flatnonzero(finite_test & (test_gc <= low_threshold)),
        )
    raise ValueError(f"Unknown GC-shift direction: {direction}")


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


def run_probe_controls_for_task(
    task: str,
    config: PipelineConfig = DEFAULT_CONFIG,
    layer: int | None = None,
) -> list[ProbeControlResult]:
    """Run probe-interpretation controls for one cached task.

    These controls test whether high probe scores can be explained by simple
    sequence composition, genomic-coordinate metadata, label leakage, or a
    train/test shift in GC composition rather than residual-stream information.
    """

    task = canonicalize_task_name(task, config.data)
    probe_layer = config.data.probe_layer if layer is None else layer
    train_payload = _load_activation_file(config.paths.activations_dir / f"{task}_train_residual_mean.npz")
    test_payload = _load_activation_file(config.paths.activations_dir / f"{task}_test_residual_mean.npz")
    x_train, y_train = _features_for_layer(train_payload, probe_layer)
    x_test, y_test = _features_for_layer(test_payload, probe_layer)
    train_sequences = train_payload["sequences"]
    test_sequences = test_payload["sequences"]
    train_names = train_payload["names"]
    test_names = test_payload["names"]

    results: list[ProbeControlResult] = []
    progress(f"Running probe controls for {task}: layer={probe_layer}")

    gc_metrics = _fit_evaluate_classifier(
        _gc_content_features(train_sequences),
        y_train,
        _gc_content_features(test_sequences),
        y_test,
        seed=config.data.seed,
    )
    results.append(
        _control_result_from_metrics(
            task,
            "gc_content_only",
            probe_layer,
            gc_metrics,
            y_train,
            y_test,
            notes="logistic control using GC fraction, GC skew, and sequence length only",
        )
    )

    train_position, train_position_coverage = _position_features(train_names)
    test_position, test_position_coverage = _position_features(test_names)
    train_valid = np.isfinite(train_position).all(axis=1)
    test_valid = np.isfinite(test_position).all(axis=1)
    if train_position_coverage >= 0.80 and test_position_coverage >= 0.80:
        position_metrics = _fit_evaluate_classifier(
            train_position[train_valid],
            y_train[train_valid],
            test_position[test_valid],
            y_test[test_valid],
            seed=config.data.seed,
        )
        results.append(
            _control_result_from_metrics(
                task,
                "position_only",
                probe_layer,
                position_metrics,
                y_train[train_valid],
                y_test[test_valid],
                notes="logistic control using parsed chromosome/start/end/midpoint/length metadata only",
            )
        )
    else:
        results.append(
            _nan_control_result(
                task,
                "position_only",
                probe_layer,
                y_train,
                y_test,
                notes=(
                    "skipped because fewer than 80% of cached names contained genomic coordinates "
                    f"(train={train_position_coverage:.3f}, test={test_position_coverage:.3f})"
                ),
            )
        )

    matched_indices = _matched_gc_indices(test_sequences, y_test, seed=config.data.seed)
    if matched_indices.size:
        matched_residual_metrics = _fit_evaluate_classifier(
            x_train,
            y_train,
            x_test[matched_indices],
            y_test[matched_indices],
            seed=config.data.seed,
        )
        results.append(
            _control_result_from_metrics(
                task,
                "residual_probe_gc_matched_test",
                probe_layer,
                matched_residual_metrics,
                y_train,
                y_test[matched_indices],
                notes="main residual probe evaluated on a balanced nearest-GC positive/negative test subset",
            )
        )
        matched_gc_metrics = _fit_evaluate_classifier(
            _gc_content_features(train_sequences),
            y_train,
            _gc_content_features(test_sequences[matched_indices]),
            y_test[matched_indices],
            seed=config.data.seed,
        )
        results.append(
            _control_result_from_metrics(
                task,
                "gc_content_only_gc_matched_test",
                probe_layer,
                matched_gc_metrics,
                y_train,
                y_test[matched_indices],
                notes="GC-only baseline evaluated on the same nearest-GC matched test subset",
            )
        )
    else:
        results.append(
            _nan_control_result(
                task,
                "residual_probe_gc_matched_test",
                probe_layer,
                y_train,
                y_test,
                notes="skipped because a balanced finite-GC matched test subset could not be built",
            )
        )
        results.append(
            _nan_control_result(
                task,
                "gc_content_only_gc_matched_test",
                probe_layer,
                y_train,
                y_test,
                notes="skipped because a balanced finite-GC matched test subset could not be built",
            )
        )

    rng = np.random.default_rng(config.data.seed)
    for run_index in range(config.data.probe_control_random_label_runs):
        shuffled_train_labels = rng.permutation(y_train)
        random_label_metrics = _fit_evaluate_classifier(
            x_train,
            shuffled_train_labels,
            x_test,
            y_test,
            seed=config.data.seed + run_index + 1,
        )
        results.append(
            _control_result_from_metrics(
                task,
                "random_label_residual_probe",
                probe_layer,
                random_label_metrics,
                shuffled_train_labels,
                y_test,
                run=run_index,
                notes="residual probe trained on shuffled train labels and evaluated against true test labels",
            )
        )

    for direction in ("low_to_high", "high_to_low"):
        train_indices, test_indices = _gc_shift_indices(train_sequences, test_sequences, direction)
        shift_metrics = _fit_evaluate_classifier(
            x_train[train_indices],
            y_train[train_indices],
            x_test[test_indices],
            y_test[test_indices],
            seed=config.data.seed,
        )
        results.append(
            _control_result_from_metrics(
                task,
                f"residual_probe_gc_shift_{direction}",
                probe_layer,
                shift_metrics,
                y_train[train_indices],
                y_test[test_indices],
                notes=(
                    "distribution-shift control: train and test on opposite sides of "
                    "the train-set GC-content 40/60 percentile split"
                ),
            )
        )

    return results


def run_probe_controls(config: PipelineConfig = DEFAULT_CONFIG) -> Path:
    """Run probe controls for all configured tasks and save a metrics table."""

    config.ensure_paths()
    results: list[ProbeControlResult] = []
    for task in config.data.task_names:
        results.extend(run_probe_controls_for_task(task, config=config))

    table = pd.DataFrame([result.__dict__ for result in results])
    output_path = config.paths.tables_dir / "linear_probe_controls.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress(f"Writing probe-control metrics table: {output_path}")
    table.to_csv(output_path, index=False)
    write_json(
        config.paths.manifests_dir / "linear_probe_controls_manifest.json",
        {
            "created_at": utc_now_iso(),
            "path": str(output_path),
            "layer": int(config.data.probe_layer),
            "tasks": list(config.data.task_names),
            "random_label_runs": int(config.data.probe_control_random_label_runs),
            "suggested_by": {
                "name": "Kiho Park",
                "url": "https://kihopark.github.io/",
                "feedback": (
                    "Clarify that linear probes establish decodability, not causality, "
                    "and add controls for correlated distributional signals."
                ),
            },
            "controls": [
                "gc_content_only",
                "position_only",
                "residual_probe_gc_matched_test",
                "gc_content_only_gc_matched_test",
                "random_label_residual_probe",
                "residual_probe_gc_shift_low_to_high",
                "residual_probe_gc_shift_high_to_low",
            ],
        },
    )
    return output_path
