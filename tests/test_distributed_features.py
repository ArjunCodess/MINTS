import numpy as np
import pandas as pd

from src.config import DataConfig, PipelineConfig, ProjectPaths
from src.distributed_features import (
    SAETrainingResult,
    _centered_cosine,
    _combined_top_feature_table,
    _rank_sae_features,
    train_sae_and_rank_features,
)


def tmp_config(tmp_path) -> PipelineConfig:
    paths = ProjectPaths(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        hf_downstream_dir=tmp_path / "data" / "hf_downstream",
        encode_dir=tmp_path / "data" / "encode" / "ctcf_gm12878",
        ctcf_dir=tmp_path / "data" / "ctcf",
        manifests_dir=tmp_path / "results" / "manifests",
        activations_dir=tmp_path / "results" / "activations",
        circuits_dir=tmp_path / "results" / "circuits",
        enrichment_dir=tmp_path / "results" / "enrichment",
        qk_alignment_dir=tmp_path / "results" / "qk_alignment",
        counterfactuals_dir=tmp_path / "results" / "counterfactuals",
        patching_dir=tmp_path / "results" / "patching",
        distributed_features_dir=tmp_path / "results" / "distributed_features",
        cross_model_dir=tmp_path / "results" / "cross_model",
        figures_dir=tmp_path / "results" / "figures",
        tables_dir=tmp_path / "results" / "tables",
        encode_url_file=tmp_path / "data" / "ENCODE4_v1.5.1_GRCh38.txt",
        grch38_fasta_gz=tmp_path / "data" / "genomes" / "hg38.fa.gz",
        grch38_fasta=tmp_path / "data" / "genomes" / "hg38.fa",
    )
    data = DataConfig(
        sae_dictionary_size=4,
        sae_epochs=1,
        sae_batch_size=4,
        sae_learning_rate=1e-2,
        sae_l1_coefficient=1e-4,
    )
    return PipelineConfig(paths=paths, data=data)


def test_centered_cosine_detects_alignment() -> None:
    assert _centered_cosine(np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0, 4.0])) > 0.99
    assert _centered_cosine(np.array([0.0, 1.0, 2.0]), np.array([2.0, 1.0, 0.0])) < -0.99


def test_rank_sae_features_sorts_by_ctcf_alignment() -> None:
    motif = np.array([0.0, 1.0, 2.0, 3.0])
    features = np.column_stack([motif, motif[::-1], np.ones_like(motif)])

    table = _rank_sae_features(features, motif, source="toy")

    assert int(table.iloc[0]["feature"]) == 0
    assert table.iloc[0]["ctcf_motif_cosine"] > 0.99


def test_train_sae_and_rank_features_writes_artifacts(tmp_path) -> None:
    config = tmp_config(tmp_path)
    activations = np.random.default_rng(0).normal(size=(12, 6)).astype(np.float32)
    motif = np.linspace(0.0, 1.0, 12, dtype=np.float32)

    result = train_sae_and_rank_features(activations, motif, "toy", config=config)

    assert result.checkpoint_path.exists()
    assert result.history_path.exists()
    assert result.alignment_path.exists()
    assert result.figure_path.exists()
    assert result.summary["dictionary_size"] == 4


def test_combined_top_feature_table_returns_global_top_10(tmp_path) -> None:
    residual_path = tmp_path / "residual.csv"
    mlp_path = tmp_path / "mlp.csv"
    pd.DataFrame(
        {
            "source": ["residual"] * 12,
            "feature": list(range(12)),
            "ctcf_motif_cosine": np.linspace(0.0, 0.11, 12),
        }
    ).to_csv(residual_path, index=False)
    pd.DataFrame(
        {
            "source": ["mlp"] * 12,
            "feature": list(range(12)),
            "ctcf_motif_cosine": np.linspace(0.2, 0.31, 12),
        }
    ).to_csv(mlp_path, index=False)
    results = [
        SAETrainingResult("residual", tmp_path / "r.pt", tmp_path / "r_hist.csv", residual_path, tmp_path / "r.png", {}),
        SAETrainingResult("mlp", tmp_path / "m.pt", tmp_path / "m_hist.csv", mlp_path, tmp_path / "m.png", {}),
    ]

    table = _combined_top_feature_table(results, top_n=10)

    assert len(table) == 10
    assert set(table["source"]) == {"mlp"}
    assert table["ctcf_motif_cosine"].is_monotonic_decreasing
