import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.native_endpoint import NativeProtocol, prepare_target, native_score, summarize_pilot, validate_loading_info
from src.utils import sha256_file
from tools.run_native_endpoint import discovery_population


class Tokenizer:
    mask_token_id = 9
    all_special_ids = [0, 9]

    def __call__(self, sequence, **kwargs):
        return dict(input_ids=[0] + ["ACGT".index(b) + 1 for b in sequence] + [0],
                    offset_mapping=[(0, 0)] + [(i, i + 1) for i in range(len(sequence))] + [(0, 0)])


def pair():
    clean = "ACGTACGTACGTACGT"
    return dict(motif=dict(clean_sequence=clean, corrupted_sequence="AGCT" + clean[4:], start=1, end=3),
                sham=dict(clean_sequence=clean, corrupted_sequence=clean[:9] + "GC" + clean[11:], start=9, end=11))


def test_target_survives_masking_with_same_span_and_id():
    target = prepare_target(Tokenizer(), pair(), NativeProtocol())
    assert target["span"] == [4, 5]
    assert target["token_id"] == 1
    assert target["nucleotides"] == "A"
    assert all(ids[target["index"]] == 9 for ids in target["masked_ids"])
    assert target["masked_ids"][0] != target["masked_ids"][1]


def test_equal_token_counts_do_not_bypass_boundary_checks():
    class Shifted(Tokenizer):
        def __call__(self, sequence, **kwargs):
            encoded = super().__call__(sequence, **kwargs)
            if sequence.startswith("AG"):
                encoded["offset_mapping"][1:3] = [(0, 2), (2, 2)]
            return encoded
    with pytest.raises(ValueError, match="correspondence"):
        prepare_target(Shifted(), pair(), NativeProtocol())


def test_sham_cannot_touch_target_or_change_composition():
    clean = "ACGTCGGTACGTACGT"
    example = dict(motif=dict(clean_sequence=clean, corrupted_sequence="AGCT" + clean[4:], start=1, end=3),
                   sham=dict(clean_sequence=clean, corrupted_sequence=clean[:4] + "GC" + clean[6:], start=4, end=6))
    assert prepare_target(Tokenizer(), example, NativeProtocol())["span"] == [6, 7]
    example["sham"]["corrupted_sequence"] = "A" * 16
    with pytest.raises(ValueError, match="composition"):
        prepare_target(Tokenizer(), example, NativeProtocol())


def test_missing_prediction_weights_are_fatal():
    with pytest.raises(ValueError, match="missing_keys"):
        validate_loading_info(dict(missing_keys=["cls.predictions.decoder.weight"]))
    with pytest.raises(ValueError, match="mismatched_keys"):
        validate_loading_info(dict(mismatched_keys=["decoder"]))
    validate_loading_info(dict(missing_keys=[], mismatched_keys=[], error_msgs=[]))


def test_native_score_uses_fixed_token_log_probability():
    class MLM:
        def __call__(self, input_ids, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[[0., 1., 2.], [2., 0., 1.]]]))
    bundle = SimpleNamespace(device="cpu", hf_model=MLM())
    score = native_score(bundle, [9, 0], dict(index=0, token_id=2))
    assert score["log_probability"] == pytest.approx(float(torch.log_softmax(torch.tensor([0., 1., 2.]), 0)[2]))
    assert score["rank"] == 1


def test_discovery_filter_excludes_inspected_confirmation_chromosomes():
    table = pd.DataFrame(dict(chrom=["chr18", "chr19", "chr20", "chr21"], start=[0]*4,
                              end=[4]*4, sequence=["ACGT"]*4))
    result = discovery_population(table, NativeProtocol())
    assert set(result.chrom) == {"chr18", "chr19"}


def pilot_rows(effects):
    return [dict(sequence_id=f"chr18:{i*2000000}-{i*2000000+16}",
                 clean_sequence=f"ACGT{i}", motif_loss=float(x), sham_loss=0.) for i, x in enumerate(effects)]


def test_feasibility_stops_without_directional_signal_and_without_clusters():
    protocol = replace(NativeProtocol(), bootstrap_samples=100, permutations=99)
    assert summarize_pilot(pilot_rows([-0.1]*12), protocol)["status"] == "stop"
    assert summarize_pilot(pilot_rows([0.2]*12), protocol)["status"] == "eligible_for_discovery_intervention"
    rows = pilot_rows([0.2]*12)
    for row in rows:
        row["sequence_id"] = "chr18:0-16"
    assert "clusters" in summarize_pilot(rows, protocol)["reasons"][0]


@pytest.mark.artifact
def test_saved_native_pilot_hashes_and_gate_reproduce():
    root = Path(__file__).resolve().parents[1]
    output = root / "results/native_endpoint"
    receipt = json.loads((output / "execution.json").read_text())
    assert receipt["status"] == "completed"
    assert not receipt["source_changed_during_run"]
    assert receipt["protocol_sha256"] == sha256_file(output / "protocol.json")
    for name, digest in receipt["source_sha256"].items():
        assert sha256_file(root / name) == digest
    for name, digest in receipt["artifacts"].items():
        assert sha256_file(output / name) == digest
    protocol = json.loads((output / "protocol.json").read_text())
    fields = asdict(NativeProtocol())
    values = {k: tuple(protocol[k]) if isinstance(v, tuple) else protocol[k] for k, v in fields.items()}
    rows = pd.read_csv(output / "scores.csv").to_dict("records")
    expected = summarize_pilot(rows, NativeProtocol(**values))
    observed = json.loads((output / "summary.json").read_text())
    assert expected["status"] == observed["status"]
    assert expected["mean"] == pytest.approx(observed["mean"])
    assert expected["ci_low"] == pytest.approx(observed["ci_low"])
    assert expected["ci_high"] == pytest.approx(observed["ci_high"])
    assert expected["genomic_clusters"] == observed["genomic_clusters"]
    assert observed["selected_head"] is None
    presentation = json.loads((output / "paper_manifest.json").read_text())
    assert presentation["tex_sha256"] == sha256_file(root / "paper/native_endpoint.tex")
    assert presentation["generator_sha256"] == sha256_file(root / "tools/build_native_artifacts.py")
    assert presentation["execution_sha256"] == sha256_file(output / "execution.json")
