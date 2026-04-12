import numpy as np

from src.enrichment import MotifSupport, attention_enrichment_ratio, matched_attention_enrichment_from_records


def test_attention_enrichment_ratio_shape_and_values() -> None:
    attention = np.ones((1, 2, 3, 4, 4), dtype=np.float32)
    attention = attention / attention.sum(axis=-1, keepdims=True)
    supports = [MotifSupport(sequence_index=0, token_start=1, token_end=3, motif_name="toy")]

    table = attention_enrichment_ratio(attention, supports)

    assert len(table) == 6
    assert set(table.columns) == {
        "layer",
        "head",
        "support_tokens",
        "attention_mass",
        "expected_mass",
        "enrichment_ratio",
    }
    assert np.allclose(table["enrichment_ratio"], 1.0)


def test_attention_enrichment_rejects_wrong_rank() -> None:
    attention = np.ones((1, 2, 3), dtype=np.float32)

    try:
        attention_enrichment_ratio(attention, [])
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for wrong attention rank.")


def test_matched_attention_enrichment_flags_motif_focused_head() -> None:
    attention_by_layer = {
        0: [
            np.asarray(
                [
                    [[0.1, 0.7, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]],
                    [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
                ],
                dtype=np.float64,
            )
        ]
    }
    supports = [MotifSupport(sequence_index=0, token_start=1, token_end=2, motif_name="toy")]

    table = matched_attention_enrichment_from_records(attention_by_layer, supports, threshold=2.0)

    head0 = table[table["head"] == 0].iloc[0]
    head1 = table[table["head"] == 1].iloc[0]
    assert head0["rho"] > 2.0
    assert bool(head0["passes_attention_enrichment"])
    assert np.isclose(head1["rho"], 1.0)
    assert not bool(head1["passes_attention_enrichment"])
