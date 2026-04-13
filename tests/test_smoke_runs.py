import numpy as np

from src.config import PipelineConfig, ProjectPaths
from src.patching import save_restoration_matrix


def test_smoke_artifacts_can_be_written_under_test_runs(tmp_path) -> None:
    test_runs = tmp_path / "results" / "test_runs"
    paths = ProjectPaths(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        results_dir=test_runs,
        hf_downstream_dir=tmp_path / "data" / "hf_downstream",
        encode_dir=tmp_path / "data" / "encode" / "ctcf_gm12878",
        ctcf_dir=tmp_path / "data" / "ctcf",
        manifests_dir=test_runs / "manifests",
        activations_dir=test_runs / "activations",
        circuits_dir=test_runs / "circuits",
        enrichment_dir=test_runs / "enrichment",
        qk_alignment_dir=test_runs / "qk_alignment",
        counterfactuals_dir=test_runs / "counterfactuals",
        patching_dir=test_runs / "patching",
        distributed_features_dir=test_runs / "distributed_features",
        cross_model_dir=test_runs / "cross_model",
        figures_dir=test_runs / "figures",
        tables_dir=test_runs / "tables",
        encode_url_file=tmp_path / "data" / "ENCODE4_v1.5.1_GRCh38.txt",
        grch38_fasta_gz=tmp_path / "data" / "genomes" / "hg38.fa.gz",
        grch38_fasta=tmp_path / "data" / "genomes" / "hg38.fa",
    )
    config = PipelineConfig(paths=paths)

    outputs = save_restoration_matrix(
        np.array([[0.0, 0.25], [0.75, np.nan]], dtype=np.float32),
        "smoke_activation_patching",
        config=config,
    )

    assert outputs["table"].is_relative_to(test_runs)
    assert outputs["figure"].is_relative_to(test_runs)
    assert outputs["manifest"].is_relative_to(test_runs)
    assert outputs["table"].exists()
    assert outputs["figure"].exists()
    assert outputs["manifest"].exists()
