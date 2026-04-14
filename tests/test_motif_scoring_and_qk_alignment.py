from pathlib import Path

import numpy as np

from src.config import PipelineConfig, ProjectPaths
from src.motif_scoring import (
    load_jaspar_ctcf_motif,
    motif_support_spans_from_start_scores,
    motif_supports_from_scores,
    score_sequence_tokens,
    token_indices_overlapping_char_span,
    token_offsets_for_sequence,
)
from src.qk_alignment import AlignmentThresholds, pearson_correlation, qk_alignment_table, qk_attention_maps, qk_key_scores


class ToyOffsetTokenizer:
    def __call__(self, sequence: str, **_kwargs):
        offsets = [(0, 0)]
        offsets.extend((idx, idx + 1) for idx in range(len(sequence)))
        offsets.append((0, 0))
        return {"offset_mapping": offsets}


class ToyKmerTokenizer:
    def __call__(self, sequence: str, **_kwargs):
        tokens = ["<cls>"]
        tokens.extend(sequence[idx : idx + 6] for idx in range(0, len(sequence), 6))
        return {"input_ids": list(range(len(tokens)))}

    def convert_ids_to_tokens(self, input_ids):
        token_values = ["<cls>", "ACGTAC", "GTACGT", "AC"]
        return [token_values[idx] for idx in input_ids]


def tmp_config(tmp_path: Path) -> PipelineConfig:
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
        figures_dir=tmp_path / "results" / "figures",
        tables_dir=tmp_path / "results" / "tables",
        encode_url_file=tmp_path / "data" / "ENCODE4_v1.5.1_GRCh38.txt",
        grch38_fasta_gz=tmp_path / "data" / "genomes" / "hg38.fa.gz",
        grch38_fasta=tmp_path / "data" / "genomes" / "hg38.fa",
    )
    return PipelineConfig(paths=paths)


def write_toy_jaspar(config: PipelineConfig) -> Path:
    jaspar_dir = config.paths.data_dir / "toy_jaspar"
    jaspar_dir.mkdir(parents=True)
    path = jaspar_dir / "MA0139.1.jaspar"
    path.write_text(
        ">MA0139.1\tCTCF\n"
        "A  [ 10 0 0 ]\n"
        "C  [ 0 10 0 ]\n"
        "G  [ 0 0 10 ]\n"
        "T  [ 1 1 1 ]\n",
        encoding="utf-8",
    )
    return path


def test_jaspar_ctcf_scores_align_to_model_tokens(tmp_path) -> None:
    config = tmp_config(tmp_path)
    path = write_toy_jaspar(config)
    motif = load_jaspar_ctcf_motif(path=path, config=config)

    record = score_sequence_tokens("TTTACGTTT", ToyOffsetTokenizer(), motif, threshold=-1e9)

    assert len(record.token_scores) == 11
    assert np.isnan(record.token_scores[0])
    assert np.isnan(record.token_scores[-1])
    assert record.support_tokens
    assert min(record.support_tokens) > 0
    assert record.support_spans
    assert all(span.token_start < span.token_end for span in record.support_spans)


def test_bpe_token_overlap_maps_full_motif_span_to_intersecting_tokens() -> None:
    offsets = [(0, 0), (0, 5), (5, 12), (12, 20), (20, 25), (25, 30)]

    overlapping = token_indices_overlapping_char_span(offsets, char_start=4, char_end=23)

    assert overlapping == [1, 2, 3, 4]


def test_motif_supports_use_bpe_intervals_not_single_tokens() -> None:
    offsets = [(0, 0), (0, 5), (5, 12), (12, 20), (20, 25), (25, 30)]
    start_scores = np.full(30, -np.inf)
    start_scores[4] = 9.0

    support_spans = motif_support_spans_from_start_scores(
        start_scores=start_scores,
        motif_length=19,
        token_offsets=offsets,
        threshold=5.0,
    )

    assert len(support_spans) == 1
    assert support_spans[0].motif_start == 4
    assert support_spans[0].motif_end == 23
    assert support_spans[0].token_start == 1
    assert support_spans[0].token_end == 5

    record = type(
        "Record",
        (),
        {"sequence_index": 0, "support_spans": support_spans},
    )()
    supports = motif_supports_from_scores([record])

    assert len(supports) == 1
    assert supports[0].token_start == 1
    assert supports[0].token_end == 5


def test_token_offsets_for_sequence_falls_back_to_fixed_kmers() -> None:
    offsets = token_offsets_for_sequence(ToyKmerTokenizer(), "ACGTACGTACGTAC")

    assert offsets == [(0, 0), (0, 6), (6, 12), (12, 14)]


def test_qk_key_scores_and_attention_maps_are_well_formed() -> None:
    hidden = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    qk = np.ones((1, 3, 3), dtype=np.float64)

    key_scores = qk_key_scores(hidden, qk, d_head=1)
    attention = qk_attention_maps(hidden, qk, d_head=1)

    assert key_scores.shape == (1, 3)
    assert np.allclose(key_scores[0] / key_scores[0, 0], [1.0, 2.0, 3.0])
    assert attention.shape == (1, 3, 3)
    assert np.allclose(attention.sum(axis=-1), 1.0)


def test_qk_alignment_table_applies_preregistered_thresholds() -> None:
    hidden = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    qk_by_layer = np.ones((1, 1, 3, 3), dtype=np.float64)
    motif_record = type(
        "MotifRecord",
        (),
        {"token_scores": np.asarray([1.0, 2.0, 3.0], dtype=np.float64)},
    )()

    table = qk_alignment_table(
        qk_by_layer=qk_by_layer,
        layer_indices=(0,),
        hidden_inputs_by_layer={0: [hidden]},
        motif_records=[motif_record],
        d_head=1,
        thresholds=AlignmentThresholds(min_r=0.5, max_p=0.05),
    )

    row = table.iloc[0]
    assert row["pearson_r"] > 0.99
    assert row["p_value"] < 0.05
    assert bool(row["passes_qk_alignment"])


def test_pearson_correlation_rejects_degenerate_vectors() -> None:
    r_value, p_value = pearson_correlation(np.asarray([1.0, 1.0, 1.0]), np.asarray([1.0, 2.0, 3.0]))

    assert np.isnan(r_value)
    assert np.isnan(p_value)
