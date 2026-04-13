import numpy as np

from src.probing import bootstrap_probe_confidence_intervals


def test_bootstrap_probe_confidence_intervals_are_deterministic() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
    predictions = (probabilities >= 0.5).astype(int)

    first = bootstrap_probe_confidence_intervals(
        y_true,
        probabilities,
        predictions,
        seed=7,
        n_bootstraps=32,
        confidence_level=0.90,
    )
    second = bootstrap_probe_confidence_intervals(
        y_true,
        probabilities,
        predictions,
        seed=7,
        n_bootstraps=32,
        confidence_level=0.90,
    )

    assert first == second
    assert set(first) == {"auroc", "auprc", "accuracy"}
    assert first["auroc"][0] <= first["auroc"][1]
    assert first["auprc"][0] <= first["auprc"][1]
    assert first["accuracy"][0] <= first["accuracy"][1]


def test_bootstrap_probe_confidence_intervals_can_be_skipped() -> None:
    intervals = bootstrap_probe_confidence_intervals(
        np.array([0, 1]),
        np.array([0.25, 0.75]),
        np.array([0, 1]),
        seed=0,
        n_bootstraps=0,
    )

    assert np.isnan(intervals["auroc"][0])
    assert np.isnan(intervals["auprc"][0])
    assert np.isnan(intervals["accuracy"][0])
