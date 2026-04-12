"""Linear probing over frozen residual stream vectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .data_ingestion import canonicalize_task_name
from .utils import utc_now_iso, write_json


@dataclass(frozen=True)
class ProbeResult:
    """Metrics for one task probe."""

    task: str
    layer: int
    auroc: float
    auprc: float
    accuracy: float
    train_examples: int
    test_examples: int


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
    train_payload = _load_activation_file(config.paths.activations_dir / f"{task}_train_residual_mean.npz")
    test_payload = _load_activation_file(config.paths.activations_dir / f"{task}_test_residual_mean.npz")
    x_train, y_train = _features_for_layer(train_payload, probe_layer)
    x_test, y_test = _features_for_layer(test_payload, probe_layer)

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

    return ProbeResult(
        task=task,
        layer=probe_layer,
        auroc=float(roc_auc_score(y_test, probabilities)),
        auprc=float(average_precision_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        train_examples=int(len(y_train)),
        test_examples=int(len(y_test)),
    )


def run_all_probes(config: PipelineConfig = DEFAULT_CONFIG) -> Path:
    """Run logistic probes for all configured tasks and save a metrics table."""

    config.ensure_paths()
    results = [train_logistic_probe_for_task(task, config=config) for task in config.data.task_names]
    table = pd.DataFrame([result.__dict__ for result in results])
    output_path = config.paths.tables_dir / "linear_probe_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
