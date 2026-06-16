"""Downstream task-performance context before mechanistic claims."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .data_ingestion import canonicalize_task_name
from .probing import _gc_content_features
from .utils import progress, utc_now_iso


@dataclass(frozen=True)
class TaskPerformanceResult:
    """Raw-sequence baselines and frozen-readout context for one task."""

    task: str
    train_examples: int
    test_examples: int
    gc_only_auroc: float
    gc_only_auprc: float
    gc_only_accuracy: float
    kmer_tfidf_3_6_auroc: float
    kmer_tfidf_3_6_auprc: float
    kmer_tfidf_3_6_accuracy: float
    frozen_dnabert_l11_readout_auroc: float
    frozen_dnabert_l11_readout_auprc: float
    frozen_dnabert_l11_readout_accuracy: float
    sequence_classifier_status: str
    notes: str


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Return binary classification metrics from probability scores."""

    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

    predictions = (probabilities >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
    }


def _fit_gc_baseline(
    train_sequences: list[str],
    y_train: np.ndarray,
    test_sequences: list[str],
    y_test: np.ndarray,
    seed: int,
):
    """Fit the same simple composition baseline used by probe controls."""

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    model.fit(_gc_content_features(np.asarray(train_sequences)), y_train)
    probabilities = model.predict_proba(_gc_content_features(np.asarray(test_sequences)))[:, 1]
    return _classification_metrics(y_test, probabilities)


def _fit_kmer_baseline(
    train_sequences: list[str],
    y_train: np.ndarray,
    test_sequences: list[str],
    y_test: np.ndarray,
    seed: int,
):
    """Fit a reproducible 3-6-mer TF-IDF logistic baseline."""

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 6),
            lowercase=False,
            min_df=2,
            max_features=50000,
            dtype=np.float32,
        ),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, solver="saga"),
    )
    model.fit(train_sequences, y_train)
    probabilities = model.predict_proba(test_sequences)[:, 1]
    return _classification_metrics(y_test, probabilities)


def _load_probe_readout_metrics(config: PipelineConfig) -> dict[str, dict[str, float]]:
    """Load frozen DNABERT readout metrics when the probe table already exists."""

    probe_path = config.paths.tables_dir / "linear_probe_metrics.csv"
    if not probe_path.exists():
        return {}
    table = pd.read_csv(probe_path)
    return {
        str(row.task): {
            "auroc": float(row.auroc),
            "auprc": float(row.auprc),
            "accuracy": float(row.accuracy),
        }
        for row in table.itertuples(index=False)
    }


def evaluate_task_performance_context(config: PipelineConfig = DEFAULT_CONFIG) -> Path:
    """Write the downstream-performance context table.

    The table intentionally separates raw-sequence task baselines from frozen
    residual readouts. The latter are useful context, but they remain diagnostic
    decodability evidence rather than end-to-end fine-tuned model performance.
    """

    from datasets import load_from_disk

    config.ensure_paths()
    readout_metrics = _load_probe_readout_metrics(config)
    rows: list[TaskPerformanceResult] = []

    for task_name in config.data.task_names:
        task = canonicalize_task_name(task_name, config.data)
        task_path = config.paths.hf_downstream_dir / task
        progress(f"Evaluating downstream task baselines: {task}")
        dataset = load_from_disk(str(task_path))
        train_sequences = list(dataset["train"]["sequence"])
        test_sequences = list(dataset["test"]["sequence"])
        y_train = np.asarray(dataset["train"]["label"], dtype=int)
        y_test = np.asarray(dataset["test"]["label"], dtype=int)

        gc = _fit_gc_baseline(train_sequences, y_train, test_sequences, y_test, seed=config.data.seed)
        kmer = _fit_kmer_baseline(train_sequences, y_train, test_sequences, y_test, seed=config.data.seed)
        readout = readout_metrics.get(task, {})
        rows.append(
            TaskPerformanceResult(
                task=task,
                train_examples=int(len(y_train)),
                test_examples=int(len(y_test)),
                gc_only_auroc=gc["auroc"],
                gc_only_auprc=gc["auprc"],
                gc_only_accuracy=gc["accuracy"],
                kmer_tfidf_3_6_auroc=kmer["auroc"],
                kmer_tfidf_3_6_auprc=kmer["auprc"],
                kmer_tfidf_3_6_accuracy=kmer["accuracy"],
                frozen_dnabert_l11_readout_auroc=float(readout.get("auroc", np.nan)),
                frozen_dnabert_l11_readout_auprc=float(readout.get("auprc", np.nan)),
                frozen_dnabert_l11_readout_accuracy=float(readout.get("accuracy", np.nan)),
                sequence_classifier_status="not_run_in_current_artifact",
                notes=(
                    "GC and k-mer are raw-sequence baselines; frozen DNABERT readout is residual "
                    "decodability, not end-to-end fine-tuned task performance."
                ),
            )
        )

    output_path = config.paths.tables_dir / "downstream_task_performance.csv"
    pd.DataFrame([row.__dict__ for row in rows]).to_csv(output_path, index=False)
    manifest = {
        "created_at": utc_now_iso(),
        "path": str(output_path),
        "tasks": [row.task for row in rows],
        "baselines": {
            "gc_only": "logistic regression over GC fraction, GC skew, and sequence length",
            "kmer_tfidf_3_6": "TF-IDF character 3-6-mer logistic regression with max_features=50000",
            "frozen_dnabert_l11_readout": (
                "existing layer-11 residual readout from linear_probe_metrics.csv; "
                "reported separately from task baselines"
            ),
        },
        "sequence_classifier_status": "not_run_in_current_artifact",
    }
    (config.paths.manifests_dir / "downstream_task_performance_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return output_path
