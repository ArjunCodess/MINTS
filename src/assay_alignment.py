"""Strict nucleotide correspondence and logged intervention position schemes."""
from __future__ import annotations

import numpy as np


def exact_offsets(tokenizer, sequence):
    encoded = tokenizer(sequence, return_offsets_mapping=True, add_special_tokens=True, truncation=False)
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("Exact tokenizer offsets are required for scientific interventions")
    offsets = [(int(a), int(b)) for a, b in offsets]
    real = [(a, b) for a, b in offsets if b > a]
    if not real or real[0][0] != 0 or real[-1][1] != len(sequence):
        raise ValueError("Tokenizer offsets do not cover the entire nucleotide sequence")
    if any(a < 0 or b < a or b > len(sequence) for a, b in offsets):
        raise ValueError("Invalid tokenizer interval")
    if any(a != previous_b for (_, previous_b), (a, _) in zip(real, real[1:])):
        raise ValueError("Tokenizer offsets contain gaps or overlapping intervals")
    if "input_ids" in encoded and len(encoded["input_ids"]) != len(offsets):
        raise ValueError("Token IDs and nucleotide offsets differ in length")
    return offsets


def validate_patch_alignment(tokenizer, clean, corrupted, positions=None):
    clean_offsets, corrupted_offsets = exact_offsets(tokenizer, clean), exact_offsets(tokenizer, corrupted)
    if len(clean) != len(corrupted) or len(clean_offsets) != len(corrupted_offsets):
        raise ValueError("Clean and corrupted sequences have incompatible nucleotide/token lengths")
    positions = list(range(len(clean_offsets))) if positions is None else list(positions)
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("Patch positions must be nonempty and unique")
    if any(i < 0 or i >= len(clean_offsets) for i in positions):
        raise ValueError("Patch position is outside the token sequence")
    mismatches = [i for i in positions if clean_offsets[i] != corrupted_offsets[i]]
    if mismatches:
        raise ValueError(f"Nucleotide boundaries differ at patch positions: {mismatches}")
    return dict(clean_offsets=clean_offsets, corrupted_offsets=corrupted_offsets, positions=positions,
                correspondence="identical half-open nucleotide intervals at every patched index")


def intervention_positions(offsets, span, scheme="edit", seed=1729, budget=None, flank_bp=20):
    start, end = span
    if not (0 <= start < end <= max(b for _, b in offsets)):
        raise ValueError("Invalid intervention span")
    real = [i for i, (a, b) in enumerate(offsets) if b > a]
    edit = [i for i in real if offsets[i][0] < end and offsets[i][1] > start]
    if not edit:
        raise ValueError("Edit span lacks token support")
    if scheme == "edit":
        chosen = edit
    elif scheme in {"all", "no_special"}:
        chosen = real
    elif scheme == "all_with_special":
        chosen = list(range(len(offsets)))
    elif scheme == "flank":
        chosen = [i for i in real if i not in edit and offsets[i][0] < end + flank_bp and offsets[i][1] > start - flank_bp]
    elif scheme == "random":
        background = [i for i in real if i not in edit]
        count = len(edit) if budget is None else budget
        if count < 1 or len(background) < count:
            raise ValueError("Insufficient positions for an equal-budget random intervention")
        chosen = np.random.default_rng(seed).choice(background, count, replace=False).tolist()
    elif scheme == "sparse":
        from .patching import stream_sparse_patch_positions
        chosen = [i for i in stream_sparse_patch_positions(len(offsets), (min(edit), max(edit) + 1),
                  max_positions=budget or max(8, 4 * int(np.ceil(np.log2(len(offsets)))))) if i in real]
    else:
        raise ValueError(f"Unknown intervention scheme: {scheme}")
    if not chosen:
        raise ValueError("Intervention scheme selected no nucleotide positions")
    return sorted(chosen)


def base_attention_density(attention, offsets, target_span, query_span=None):
    """Distribute each key's attention uniformly over its nucleotide interval.

    Queries are weighted by nucleotide width, rather than treating long and
    short BPE tokens as equal observations. Special-token mass remains in the
    native softmax denominator, but has no nucleotide support.
    """
    a = np.asarray(attention, dtype=float)
    offsets = np.asarray(offsets, dtype=int)
    if a.shape[-2:] != (len(offsets), len(offsets)):
        raise ValueError("Attention and offsets must align")
    widths = offsets[:, 1] - offsets[:, 0]
    start, end = target_span
    length = int(widths.sum())
    if not (0 <= start < end <= length):
        raise ValueError("Invalid target nucleotide span")
    overlap = np.maximum(0, np.minimum(offsets[:, 1], end) - np.maximum(offsets[:, 0], start))
    key_fraction = np.divide(overlap, widths, out=np.zeros(len(widths), dtype=float), where=widths > 0)
    query_weights = widths.astype(float)
    if query_span is not None:
        qs, qe = query_span
        if not (0 <= qs < qe <= length):
            raise ValueError("Invalid query nucleotide span")
        query_weights = np.maximum(0, np.minimum(offsets[:, 1], qe) - np.maximum(offsets[:, 0], qs)).astype(float)
    if query_weights.sum() == 0:
        raise ValueError("Query span lacks nucleotide support")
    return np.einsum("...ij,i,j->...", a, query_weights / query_weights.sum(), key_fraction) * length / (end - start)
