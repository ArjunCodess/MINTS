from src.config import PipelineConfig, ProjectPaths
from src.cross_model import (
    attention_enrichment_delta,
    build_cross_model_config,
    probe_metric_deltas,
)
from src.modeling import (
    DNABERT2_MODEL_NAME,
    NUCLEOTIDE_TRANSFORMER_MODEL_NAME,
    model_slug,
    tokenization_family,
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
    return PipelineConfig(paths=paths)


def test_tokenization_family_and_slug() -> None:
    assert tokenization_family(DNABERT2_MODEL_NAME) == "BPE"
    assert tokenization_family(NUCLEOTIDE_TRANSFORMER_MODEL_NAME) == "fixed_6mer"
    assert "/" not in model_slug(NUCLEOTIDE_TRANSFORMER_MODEL_NAME)


def test_cross_model_config_uses_isolated_result_paths(tmp_path) -> None:
    config = tmp_config(tmp_path)
    model_config = build_cross_model_config(config, NUCLEOTIDE_TRANSFORMER_MODEL_NAME)

    assert model_config.model.model_name == NUCLEOTIDE_TRANSFORMER_MODEL_NAME
    assert model_config.paths.hf_downstream_dir == config.paths.hf_downstream_dir
    assert model_config.paths.results_dir.parent == config.paths.cross_model_dir
    assert model_slug(NUCLEOTIDE_TRANSFORMER_MODEL_NAME) in str(model_config.paths.results_dir)


def test_probe_metric_deltas_are_comparison_minus_base() -> None:
    base = [{"task": "promoter_tata", "auroc": 0.8, "auprc": 0.7, "accuracy": 0.6}]
    comparison = [{"task": "promoter_tata", "auroc": 0.9, "auprc": 0.6, "accuracy": 0.65}]

    rows = probe_metric_deltas(base, comparison)

    by_metric = {row["metric"]: row["delta_comparison_minus_base"] for row in rows}
    assert by_metric == {"accuracy": 0.050000000000000044, "auprc": -0.09999999999999998, "auroc": 0.09999999999999998}


def test_attention_enrichment_delta_handles_missing_scores() -> None:
    delta = attention_enrichment_delta(
        {"best_rho": 2.5, "candidate_heads": 3},
        {"best_rho": None, "candidate_heads": 0},
    )

    assert delta["delta_best_rho_comparison_minus_base"] is None
    assert delta["delta_candidate_heads_comparison_minus_base"] == -3
