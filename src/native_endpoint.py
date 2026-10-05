"""Discovery-only, fixed-target masked-flank feasibility experiment.

This module never fits a readout, changes endpoints after inspection, or treats
a sequence-modeling effect as evidence of biological binding causality.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np
import torch

from .assay_alignment import exact_offsets
from .assay_stats import genomic_clusters, paired_head_inference
from .config import DEFAULT_CONFIG
from .controlled_edits import edit_signature
from .modeling import (
    LoadedModelBundle, _disable_dnabert_triton_attention,
    _patch_dnabert_alibi_builder, _patch_remote_masked_lm_class,
    _patch_runtime_compat_methods, _patch_transformers_pruning_helper,
    _rebuild_alibi_on_device, _resolve_device,
)


@dataclass(frozen=True)
class NativeProtocol:
    endpoint: str = "unchanged masked flank token log probability"
    target_rule: str = "nearest downstream whole token, gap 1..30 bp, then upstream"
    seed: int = 1729
    sequence_cap: int = 12
    scan_cap: int = 256
    edit_candidates: int = 128
    max_flank_bp: int = 30
    min_clusters: int = 8
    min_effect_nats: float = 0.05
    bootstrap_samples: int = 2000
    permutations: int = 9999
    discovery_chromosomes: tuple[str, ...] = ("chr18", "chr19")
    previously_inspected_chromosomes: tuple[str, ...] = ("chr20", "chr21")
    primary_intervention: str = "clean to motif-edited head context at masked query"
    head_selection: str = "largest mean motif-minus-sham rescue; ties by layer then head"
    claim_boundary: str = "native sequence prediction, not CTCF binding causality"

    def __post_init__(self):
        if min(self.sequence_cap, self.scan_cap, self.edit_candidates, self.max_flank_bp) < 1:
            raise ValueError("Protocol caps and flank distance must be positive")
        if self.min_clusters < 2 or self.bootstrap_samples < 2 or self.permutations < 1:
            raise ValueError("Protocol requires independent clusters and resampling")
        if not math.isfinite(self.min_effect_nats) or self.min_effect_nats <= 0:
            raise ValueError("The sensitivity threshold must be finite and positive")


def validate_loading_info(info):
    """Reject silently initialized parameters, including a random MLM head."""
    for key in ("missing_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise ValueError(f"Pretrained native endpoint has {key}: {info[key]}")


def load_native_mlm(config=DEFAULT_CONFIG.model):
    """Load only the pinned pretrained MLM, without the encoder fallback."""
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    if not config.revision or config.model_name != DEFAULT_CONFIG.model.model_name:
        raise ValueError("The native pilot requires the pinned DNABERT-2 checkpoint")
    kwargs = dict(revision=config.revision, trust_remote_code=config.trust_remote_code,
                  local_files_only=config.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, **kwargs)
    hf_config = AutoConfig.from_pretrained(config.model_name, **kwargs)
    # The pinned remote class predates Transformers 5's config defaults.
    hf_config.is_decoder = False
    hf_config.pad_token_id = tokenizer.pad_token_id
    if hf_config._commit_hash != config.revision:
        raise ValueError("Resolved checkpoint does not match the protocol revision")
    _patch_transformers_pruning_helper()
    _patch_dnabert_alibi_builder(config.model_name, hf_config, config.revision)
    _disable_dnabert_triton_attention(config.model_name, hf_config, config.revision)
    _patch_remote_masked_lm_class(config.model_name, hf_config, config.revision)
    model, info = AutoModelForMaskedLM.from_pretrained(
        config.model_name, config=hf_config, output_loading_info=True, **kwargs)
    validate_loading_info(info)
    info = {key: sorted(value) if isinstance(value, set) else value for key, value in info.items()}
    if not hasattr(model, "cls") or tokenizer.mask_token_id is None:
        raise ValueError("Checkpoint lacks the native prediction head or mask token")
    device = _resolve_device(config.device)
    _patch_runtime_compat_methods(model.bert)
    model.bert.pooler = None
    _rebuild_alibi_on_device(model.bert, device)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    audit = dict(model_name=config.model_name, revision=config.revision,
                 resolved_revision=hf_config._commit_hash, loading_info=info,
                 prediction_head_parameters=[n for n, _ in model.named_parameters() if n.startswith("cls.")],
                 fitted_readout=False, device=device)
    return LoadedModelBundle(tokenizer, model, None, config.model_name, device, "native_mlm"), audit


def prepare_target(tokenizer, pair, protocol):
    """Choose the target by sequence geometry alone, before any model scores."""
    motif, sham = pair["motif"], pair["sham"]
    clean = motif["clean_sequence"]
    sequences = [clean, motif["corrupted_sequence"], sham["corrupted_sequence"]]
    if sham["clean_sequence"] != clean or any(len(s) != len(clean) for s in sequences):
        raise ValueError("Motif/sham sequences must share the same clean sequence and length")
    if any(Counter(s) != Counter(clean) for s in sequences):
        raise ValueError("Pilot edits must preserve nucleotide composition")
    if edit_signature(clean, sequences[1]) != edit_signature(clean, sequences[2]):
        raise ValueError("Motif and sham transitions must match")
    for record in (motif, sham):
        a, b = record["start"], record["end"]
        if not (0 <= a < b <= len(clean)):
            raise ValueError("Invalid edit interval")
        if clean[:a] != record["corrupted_sequence"][:a] or clean[b:] != record["corrupted_sequence"][b:]:
            raise ValueError("Edits must remain within their recorded intervals")
    if motif["end"] - motif["start"] != sham["end"] - sham["start"]:
        raise ValueError("Motif and sham window widths must match")
    offsets = [exact_offsets(tokenizer, s) for s in sequences]
    if offsets[0] != offsets[1] or offsets[0] != offsets[2]:
        raise ValueError("Native endpoint requires exact whole-sequence token correspondence")
    ids = [list(tokenizer(s, add_special_tokens=True, truncation=False)["input_ids"]) for s in sequences]
    if any(len(row) != len(offsets[0]) for row in ids):
        raise ValueError("Token IDs do not align with nucleotide offsets")
    start, end = motif["start"], motif["end"]
    if not (0 <= start < end <= len(clean)):
        raise ValueError("Invalid motif interval")
    # Require a visible motif difference, rather than masking away the perturbation.
    motif_positions = [i for i, (a, b) in enumerate(offsets[0]) if b > a and a < end and b > start]
    if not any(ids[0][i] != ids[1][i] for i in motif_positions):
        raise ValueError("The motif difference is not visible to the tokenizer")
    candidates = []
    for i, (a, b) in enumerate(offsets[0]):
        if b <= a:
            continue
        downstream = a >= end + 1 and a - end <= protocol.max_flank_bp
        upstream = b <= start - 1 and start - b <= protocol.max_flank_bp
        if not (downstream or upstream):
            continue
        # Reject any target touched by the matched sham window.
        if a < sham["end"] and b > sham["start"]:
            continue
        if len({s[a:b] for s in sequences}) != 1 or len({row[i] for row in ids}) != 1:
            continue
        candidates.append((0 if downstream else 1, a - end if downstream else start - b, i))
    if not candidates:
        raise ValueError("No unchanged eligible flank target")
    index = min(candidates)[2]
    a, b = offsets[0][index]
    target_id = ids[0][index]
    if target_id in tokenizer.all_special_ids:
        raise ValueError("The target cannot be a special token")
    for row in ids:
        row[index] = tokenizer.mask_token_id
    return dict(index=index, token_id=target_id, span=[a, b], nucleotides=clean[a:b],
                masked_ids=ids, offsets=offsets[0], motif_visible=True)


def native_score(bundle, masked_ids, target):
    ids = torch.tensor([masked_ids], dtype=torch.long, device=bundle.device)
    with torch.no_grad():
        logits = bundle.hf_model(input_ids=ids, attention_mask=torch.ones_like(ids), return_dict=True).logits
    if logits.ndim != 3 or logits.shape[:2] != ids.shape or not torch.isfinite(logits).all():
        raise ValueError("Native MLM logits must be finite and preserve token positions")
    values = torch.log_softmax(logits[0, target["index"]].float(), dim=-1)
    token_id = target["token_id"]
    return dict(log_probability=float(values[token_id].item()),
                rank=int((values > values[token_id]).sum().item()) + 1)


def summarize_pilot(rows, protocol):
    """A failed feasibility gate cannot be reinterpreted as a precise null."""
    result = dict(status="stop", reasons=[], endpoint=protocol.endpoint,
                  retained_sequences=len(rows), selected_head=None,
                  confirmation_status="not run; fresh uninspected cohort required")
    if len(rows) < 2:
        result["reasons"].append("insufficient eligible targets")
        return result
    clusters = genomic_clusters([r["sequence_id"] for r in rows],
                                sequences=[r["clean_sequence"] for r in rows])
    result["genomic_clusters"] = int(len(np.unique(clusters)))
    if result["genomic_clusters"] < protocol.min_clusters:
        result["reasons"].append("insufficient independent genomic clusters")
        return result
    contrast = np.array([r["motif_loss"] - r["sham_loss"] for r in rows])
    inference = paired_head_inference(contrast, clusters, protocol.permutations,
                                     protocol.bootstrap_samples, protocol.seed)
    for key in ("mean", "ci_low", "ci_high", "p"):
        result[key] = float(inference[key][0])
    result["cluster_standard_deviation"] = float(np.std(
        [contrast[clusters == c].mean() for c in np.unique(clusters)], ddof=1))
    if result["mean"] < protocol.min_effect_nats or result["ci_low"] <= 0:
        result["reasons"].append("motif effect does not exceed matched sham with specified sensitivity")
    if not result["reasons"]:
        result["status"] = "eligible_for_discovery_intervention"
    result["interpretation"] = "discovery feasibility only; no native mechanism or binding claim"
    return result
