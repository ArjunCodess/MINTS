from datasets import Dataset, DatasetDict

from src.config import DataConfig, PipelineConfig, ProjectPaths
from src.task_performance import evaluate_task_performance_context


def _paths(tmp_path):
    return ProjectPaths(
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


def test_evaluate_task_performance_context_writes_table(tmp_path) -> None:
    paths = _paths(tmp_path)
    config = PipelineConfig(
        paths=paths,
        data=DataConfig(task_names=("promoter_tata",)),
    )
    config.ensure_paths()

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "sequence": [
                        "AAAAATATATAA",
                        "AAAATATATATA",
                        "ATATATAAAAAA",
                        "TATAAAAAATAT",
                        "GCGCGCGGGGCC",
                        "CCCGCGCGCGGG",
                        "GGGGCCCCGCGC",
                        "CGCGGGGCCCCC",
                    ],
                    "label": [0, 0, 0, 0, 1, 1, 1, 1],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "sequence": ["AAAATATATAAA", "ATATAAAAAAAA", "GCGCGGGGCCCC", "CCCGCGCGGGGG"],
                    "label": [0, 0, 1, 1],
                }
            ),
        }
    )
    dataset.save_to_disk(str(paths.hf_downstream_dir / "promoter_tata"))
    paths.activations_dir.mkdir(parents=True, exist_ok=True)
    train_features = [
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [0.1, 0.1],
        [2.0, 2.0],
        [2.1, 2.0],
        [2.0, 2.1],
        [2.1, 2.1],
    ]
    test_features = [[0.0, 0.0], [0.1, 0.0], [2.0, 2.0], [2.1, 2.0]]
    import numpy as np

    np.savez(
        paths.activations_dir / "promoter_tata_train_residual_mean.npz",
        residual_mean=np.asarray(train_features, dtype=np.float32)[:, None, :],
        labels=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=int),
        names=np.asarray([f"train_{idx}" for idx in range(8)]),
        sequences=np.asarray(dataset["train"]["sequence"]),
        layers=np.asarray([11], dtype=int),
    )
    np.savez(
        paths.activations_dir / "promoter_tata_test_residual_mean.npz",
        residual_mean=np.asarray(test_features, dtype=np.float32)[:, None, :],
        labels=np.asarray([0, 0, 1, 1], dtype=int),
        names=np.asarray([f"test_{idx}" for idx in range(4)]),
        sequences=np.asarray(dataset["test"]["sequence"]),
        layers=np.asarray([11], dtype=int),
    )

    table_path = evaluate_task_performance_context(config=config)
    table_text = table_path.read_text(encoding="utf-8")
    manifest_text = (paths.manifests_dir / "downstream_task_performance_manifest.json").read_text(
        encoding="utf-8"
    )

    assert "promoter_tata" in table_text
    assert "kmer_tfidf_3_6_auroc" in table_text
    assert "dnabert_sequence_head_auroc" in table_text
    assert "frozen_dnabert_l11_sequence_head" in table_text
    assert "downstream_task_performance.csv" in manifest_text
