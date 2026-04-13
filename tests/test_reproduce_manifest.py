from src.config import PipelineConfig
from src.reproduce import _load_probe_metrics


def test_load_probe_metrics_treats_blank_confidence_intervals_as_none(tmp_path) -> None:
    metrics_path = tmp_path / "linear_probe_metrics.csv"
    metrics_path.write_text(
        "\n".join(
            [
                "task,layer,auroc,auroc_ci_low,auroc_ci_high,auprc,auprc_ci_low,auprc_ci_high,accuracy,accuracy_ci_low,accuracy_ci_high,train_examples,test_examples,train_positive_rate,test_positive_rate",
                "promoter_tata,11,0.9,,0.95,0.8,,,0.7,0.6,,10,4,0.5,0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    row = _load_probe_metrics(metrics_path, PipelineConfig())[0]

    assert row["auroc_ci_low"] is None
    assert row["auroc_ci_high"] == 0.95
    assert row["auprc_ci_low"] is None
    assert row["auprc_ci_high"] is None
    assert row["accuracy_ci_low"] == 0.6
    assert row["accuracy_ci_high"] is None
