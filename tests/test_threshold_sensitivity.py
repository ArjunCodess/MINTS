import pandas as pd

from src.config import PipelineConfig, ProjectPaths
from src.threshold_sensitivity import run_threshold_sensitivity


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


def test_run_threshold_sensitivity_writes_table_and_figure(tmp_path) -> None:
    config = PipelineConfig(paths=_paths(tmp_path))
    config.ensure_paths()
    pd.DataFrame(
        [
            {"layer": 0, "head": 0, "pearson_r": 0.35, "p_value": 0.01, "n_positions": 100, "passes_qk_alignment": False},
            {"layer": 0, "head": 1, "pearson_r": 0.05, "p_value": 0.20, "n_positions": 100, "passes_qk_alignment": False},
            {"layer": 1, "head": 0, "pearson_r": 0.51, "p_value": 0.01, "n_positions": 100, "passes_qk_alignment": True},
            {"layer": 1, "head": 1, "pearson_r": -0.10, "p_value": 0.01, "n_positions": 100, "passes_qk_alignment": False},
        ]
    ).to_csv(config.paths.qk_alignment_dir / "ctcf_qk_alignment.csv", index=False)
    pd.DataFrame(
        [
            {"layer": 0, "head": 0, "support_tokens": 10, "background_tokens": 10, "a_motif": 2.0, "a_bg": 1.0, "rho": 2.0, "passes_attention_enrichment": True},
            {"layer": 0, "head": 1, "support_tokens": 10, "background_tokens": 10, "a_motif": 1.1, "a_bg": 1.0, "rho": 1.1, "passes_attention_enrichment": False},
            {"layer": 1, "head": 0, "support_tokens": 10, "background_tokens": 10, "a_motif": 1.3, "a_bg": 1.0, "rho": 1.3, "passes_attention_enrichment": False},
            {"layer": 1, "head": 1, "support_tokens": 10, "background_tokens": 10, "a_motif": 0.9, "a_bg": 1.0, "rho": 0.9, "passes_attention_enrichment": False},
        ]
    ).to_csv(config.paths.enrichment_dir / "ctcf_qk_alignment_matched_attention_enrichment.csv", index=False)
    pd.DataFrame(
        [
            {"layer": 0, "head": 0, "restoration": 0.8, "task": "promoter_tata", "pairs": 4, "denominator_failures": 0},
            {"layer": 0, "head": 1, "restoration": 0.1, "task": "promoter_tata", "pairs": 4, "denominator_failures": 0},
        ]
    ).to_csv(config.paths.patching_dir / "promoter_tata_batch_dnabert_activation_patching.csv", index=False)
    pd.DataFrame(
        [
            {"sequence_index": 0, "token_index": 0, "char_start": 0, "char_end": 4, "motif_score": 9.0, "threshold": 8.0, "is_support": True},
            {"sequence_index": 0, "token_index": 1, "char_start": 4, "char_end": 8, "motif_score": 2.0, "threshold": 8.0, "is_support": False},
            {"sequence_index": 1, "token_index": 0, "char_start": 0, "char_end": 4, "motif_score": 7.0, "threshold": 8.0, "is_support": True},
            {"sequence_index": 1, "token_index": 1, "char_start": 4, "char_end": 8, "motif_score": 1.0, "threshold": 8.0, "is_support": False},
        ]
    ).to_csv(config.paths.enrichment_dir / "ctcf_qk_alignment_token_motif_scores.csv", index=False)
    pd.DataFrame({"sequence": ["ACGTACGT", "GGGGCCCC"]}).to_csv(
        config.paths.ctcf_dir / "ctcf_gm12878_sequences.tsv",
        sep="\t",
        index=False,
    )

    outputs = run_threshold_sensitivity(config=config)
    table = pd.read_csv(outputs.table)

    assert outputs.figure.exists()
    assert outputs.manifest.exists()
    assert "joint_ctcf_sensitivity" in set(table["analysis"])
    assert "shuffled_motif_scores" in set(table["null_type"].dropna())
    assert "gc_matched_background" in set(table["null_type"].dropna())
    qk_row = table[(table["metric"] == "qk_pearson_r") & (table["r_threshold"] == 0.5)].iloc[0]
    assert int(qk_row["observed_pass_count"]) == 1
