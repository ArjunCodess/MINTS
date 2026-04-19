import numpy as np

from src.config import DataConfig, PipelineConfig, ProjectPaths
from src.probing import bootstrap_probe_confidence_intervals, run_probe_controls


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


def test_run_probe_controls_writes_control_table(tmp_path) -> None:
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
        task_names=("promoter_tata",),
        activation_layers=(11,),
        probe_layer=11,
        probe_control_random_label_runs=2,
    )
    config = PipelineConfig(paths=paths, data=data)
    config.ensure_paths()

    train_vectors = np.array(
        [
            [[0.0, 0.0]],
            [[0.1, 0.0]],
            [[1.0, 1.0]],
            [[1.1, 1.0]],
            [[0.2, 0.1]],
            [[0.9, 1.2]],
        ],
        dtype=np.float32,
    )
    test_vectors = np.array(
        [
            [[0.0, 0.1]],
            [[1.0, 1.1]],
            [[0.2, 0.0]],
            [[1.1, 0.9]],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        paths.activations_dir / "promoter_tata_train_residual_mean.npz",
        residual_mean=train_vectors,
        labels=np.array([0, 0, 1, 1, 0, 1], dtype=np.int64),
        names=np.array(
            [
                "chr1:100-120|0",
                "chr1:200-220|0",
                "chr2:300-320|1",
                "chr2:400-420|1",
                "chr3:500-520|0",
                "chr3:600-620|1",
            ]
        ),
        sequences=np.array(["ATAT", "AATT", "GCGC", "GGCC", "ATGC", "CCGG"]),
        layers=np.array([11], dtype=np.int64),
    )
    np.savez_compressed(
        paths.activations_dir / "promoter_tata_test_residual_mean.npz",
        residual_mean=test_vectors,
        labels=np.array([0, 1, 0, 1], dtype=np.int64),
        names=np.array(["chr1:700-720|0", "chr2:800-820|1", "chr3:900-920|0", "chr4:1000-1020|1"]),
        sequences=np.array(["AATA", "GCGG", "ATGC", "CCGC"]),
        layers=np.array([11], dtype=np.int64),
    )

    table_path = run_probe_controls(config=config)
    table_text = table_path.read_text(encoding="utf-8")
    manifest_text = (paths.manifests_dir / "linear_probe_controls_manifest.json").read_text(encoding="utf-8")

    assert "gc_content_only" in table_text
    assert "position_only" in table_text
    assert "residual_probe_gc_matched_test" in table_text
    assert table_text.count("random_label_residual_probe") == 2
    assert "Kiho Park" in manifest_text
