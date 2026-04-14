import math

import numpy as np

from src.config import PipelineConfig, ProjectPaths
from src.counterfactuals import char_span_to_token_span, generate_counterfactual_sequence
from src.patching import (
    _replace_concatenated_head_slice,
    hierarchical_sparse_patch_positions,
    layer_head_restoration_matrix,
    patch_head_output_tensor,
    restoration_metric,
    save_restoration_matrix,
    stream_sparse_attention_mask,
    stream_sparse_patch_positions,
)


class ToyOffsetTokenizer:
    def __call__(self, sequence: str, **_kwargs):
        offsets = [(0, 0)]
        offsets.extend((idx, idx + 1) for idx in range(len(sequence)))
        offsets.append((0, 0))
        return {"offset_mapping": offsets}


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
        counterfactuals_dir=tmp_path / "results" / "counterfactuals",
        patching_dir=tmp_path / "results" / "patching",
        figures_dir=tmp_path / "results" / "figures",
        tables_dir=tmp_path / "results" / "tables",
        encode_url_file=tmp_path / "data" / "ENCODE4_v1.5.1_GRCh38.txt",
        grch38_fasta_gz=tmp_path / "data" / "genomes" / "hg38.fa.gz",
        grch38_fasta=tmp_path / "data" / "genomes" / "hg38.fa",
    )
    return PipelineConfig(paths=paths)


def test_tata_counterfactual_preserves_length_and_coordinates() -> None:
    record = generate_counterfactual_sequence(
        "CCCTATAAAGGG",
        task="promoter_tata",
        sequence_id="toy_promoter",
    )

    assert record.start == 3
    assert record.end == 9
    assert record.clean_subsequence == "TATAAA"
    assert record.corrupted_subsequence != record.clean_subsequence
    assert len(record.clean_sequence) == len(record.corrupted_sequence)
    assert record.corrupted_sequence[record.start : record.end] == record.corrupted_subsequence


def test_splice_counterfactual_mutates_canonical_dinucleotide() -> None:
    record = generate_counterfactual_sequence("AAAAAAGTCCCCCC", task="splice_sites_donors")

    assert record.clean_subsequence == "GT"
    assert record.corrupted_subsequence == "GC"
    assert len(record.clean_sequence) == len(record.corrupted_sequence)


def test_ctcf_counterfactual_uses_jaspar_consensus_when_available(tmp_path) -> None:
    config = tmp_config(tmp_path)
    jaspar_dir = config.paths.data_dir / "toy_jaspar"
    jaspar_dir.mkdir(parents=True)
    (jaspar_dir / "MA0139.1.jaspar").write_text(
        ">MA0139.1\tCTCF\n"
        "A  [ 9 0 0 ]\n"
        "C  [ 0 8 0 ]\n"
        "G  [ 0 0 7 ]\n"
        "T  [ 1 2 3 ]\n",
        encoding="utf-8",
    )

    record = generate_counterfactual_sequence("TTTACGAAA", task="ctcf", config=config)

    assert record.start == 3
    assert record.end == 6
    assert record.clean_subsequence == "ACG"
    assert len(record.clean_sequence) == len(record.corrupted_sequence)


def test_char_span_to_token_span_uses_offsets() -> None:
    assert char_span_to_token_span("ACGTAC", 2, 5, ToyOffsetTokenizer()) == (3, 6)


def test_restoration_metric_handles_full_and_degenerate_recovery() -> None:
    assert restoration_metric(clean_logit=3.0, corrupted_logit=1.0, patched_logit=2.0) == 0.5
    assert math.isnan(restoration_metric(clean_logit=1.0, corrupted_logit=1.0, patched_logit=2.0))


def test_patch_head_output_tensor_patches_only_target_head() -> None:
    clean = np.ones((2, 3, 4, 5), dtype=np.float32)
    corrupted = np.zeros((2, 3, 4, 5), dtype=np.float32)

    patched = patch_head_output_tensor(clean, corrupted, layer=1, head=2, positions=[0, 2])

    assert patched[1, 2, 0].sum() == 5
    assert patched[1, 2, 2].sum() == 5
    assert patched[1, 2, 1].sum() == 0
    assert patched[0].sum() == 0


def test_replace_concatenated_head_slice_preserves_other_heads() -> None:
    import torch

    clean = torch.ones((3, 6), dtype=torch.float32)
    corrupted = torch.zeros((3, 6), dtype=torch.float32)

    patched = _replace_concatenated_head_slice(corrupted, clean, head_idx=1, d_head=2)

    assert torch.allclose(patched[:, :2], torch.zeros((3, 2)))
    assert torch.allclose(patched[:, 2:4], torch.ones((3, 2)))
    assert torch.allclose(patched[:, 4:], torch.zeros((3, 2)))


def test_hierarchical_sparse_patch_positions_keep_motif_and_log_context() -> None:
    positions = hierarchical_sparse_patch_positions(
        sequence_length=128,
        motif_token_span=(60, 64),
        max_positions=16,
    )

    assert 0 in positions
    assert 127 in positions
    assert set(range(60, 64)).issubset(positions)
    assert len(positions) <= 16
    assert positions == sorted(positions)


def test_stream_sparse_attention_mask_refines_to_top_key_blocks() -> None:
    key_scores = np.zeros(128, dtype=np.float64)
    key_scores[72:80] = 10.0

    sparse = stream_sparse_attention_mask(
        sequence_length=128,
        key_scores=key_scores,
        query_block_size=16,
        min_key_block_size=8,
        top_k=1,
        locality_weight=0.0,
    )

    hot_key_block = [idx for idx, block in enumerate(sparse.key_blocks) if block == (72, 80)][0]
    assert sparse.mask.shape == (8, 16)
    assert sparse.mask[:, hot_key_block].all()
    assert sparse.metadata["algorithm"] == "stream_hierarchical_block_refinement"
    assert sparse.metadata["density"] < 0.2


def test_stream_sparse_patch_positions_keeps_motif_under_cap() -> None:
    positions = stream_sparse_patch_positions(
        sequence_length=100_000,
        motif_token_span=(50_000, 50_006),
        max_positions=64,
        top_k=2,
    )

    assert set(range(50_000, 50_006)).issubset(positions)
    assert 0 in positions
    assert 99_999 in positions
    assert len(positions) <= 64
    assert positions == sorted(positions)


def test_layer_head_restoration_matrix_averages_examples() -> None:
    clean = np.array([3.0, 5.0])
    corrupted = np.array([1.0, 1.0])
    patched = np.array(
        [
            [[2.0, 3.0], [3.0, 5.0]],
            [[1.0, 1.0], [0.0, 0.0]],
        ]
    )

    matrix = layer_head_restoration_matrix(clean, corrupted, patched)

    assert matrix.shape == (2, 2)
    assert np.allclose(matrix[0], [0.5, 1.0])
    assert np.allclose(matrix[1], [0.0, -0.375])


def test_save_restoration_matrix_writes_table_and_heatmap(tmp_path) -> None:
    config = tmp_config(tmp_path)

    outputs = save_restoration_matrix(np.array([[0.0, 1.0], [0.5, np.nan]]), "toy_restore", config)

    assert outputs["table"].exists()
    assert outputs["figure"].exists()
    assert outputs["manifest"].exists()
