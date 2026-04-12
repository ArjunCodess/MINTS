"""Counterfactual sequence generation for motif-destruction experiments."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .data_ingestion import canonicalize_task_name
from .utils import utc_now_iso, write_json


TATA_RE = re.compile(r"TATA[AT]A", re.IGNORECASE)
CTCF_CORE_RE = re.compile(r"CCGCG[ACGT]GG[ACGT]GGCAG", re.IGNORECASE)
BASE_ORDER = ("A", "C", "G", "T")


@dataclass(frozen=True)
class MutationRecord:
    """Single sequence-edit record used to reproduce a counterfactual pair."""

    sequence_id: str
    task: str
    clean_sequence: str
    corrupted_sequence: str
    motif_name: str
    start: int
    end: int
    clean_subsequence: str
    corrupted_subsequence: str
    strategy: str


def _gc_balanced_replacement(length: int) -> str:
    """Return a deterministic non-motif replacement of the requested length."""

    seed = "GCGC"
    return (seed * ((length // len(seed)) + 1))[:length]


def _replace_span(sequence: str, start: int, end: int, replacement: str) -> str:
    if start < 0 or end < start or end > len(sequence):
        raise ValueError(f"Invalid replacement span [{start}, {end}) for sequence length {len(sequence)}.")
    if len(replacement) != end - start:
        raise ValueError("Counterfactual replacement must preserve sequence length.")
    return sequence[:start] + replacement + sequence[end:]


def _find_near_center(sequence: str, motif: str, window: int = 24) -> tuple[int, int] | None:
    center = len(sequence) // 2
    starts = [match.start() for match in re.finditer(re.escape(motif), sequence, flags=re.IGNORECASE)]
    if not starts:
        return None
    best = min(starts, key=lambda idx: abs((idx + len(motif) // 2) - center))
    if abs((best + len(motif) // 2) - center) > window:
        best = starts[0]
    return best, best + len(motif)


def _first_regex_span(pattern: re.Pattern[str], sequence: str) -> tuple[int, int] | None:
    match = pattern.search(sequence)
    if match is None:
        return None
    return match.start(), match.end()


def _read_jaspar_consensus(path: Path) -> str:
    """Read a JASPAR position-count matrix and return its max-count consensus."""

    counts: dict[str, list[int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(">"):
            continue
        base = stripped[0].upper()
        if base not in BASE_ORDER:
            continue
        values = [int(value) for value in re.findall(r"\d+", stripped)]
        counts[base] = values
    if set(counts) != set(BASE_ORDER):
        raise ValueError(f"JASPAR file is missing one or more A/C/G/T rows: {path}")
    motif_length = len(counts["A"])
    if any(len(counts[base]) != motif_length for base in BASE_ORDER):
        raise ValueError(f"JASPAR rows do not have a consistent width: {path}")
    consensus = []
    for idx in range(motif_length):
        consensus.append(max(BASE_ORDER, key=lambda base: counts[base][idx]))
    return "".join(consensus)


def _find_ctcf_jaspar_path(config: PipelineConfig) -> Path | None:
    candidates = sorted(config.paths.data_dir.glob("*jaspar/MA0139*.jaspar"))
    return candidates[0] if candidates else None


def _find_ctcf_span(sequence: str, config: PipelineConfig) -> tuple[int, int] | None:
    jaspar_path = _find_ctcf_jaspar_path(config)
    if jaspar_path is not None:
        consensus = _read_jaspar_consensus(jaspar_path)
        span = _find_near_center(sequence, consensus, window=len(sequence))
        if span is not None:
            return span
    return _first_regex_span(CTCF_CORE_RE, sequence)


def _fallback_center_span(sequence: str, length: int) -> tuple[int, int]:
    length = min(length, len(sequence))
    start = max(0, (len(sequence) - length) // 2)
    return start, start + length


def generate_counterfactual_sequence(
    sequence: str,
    task: str,
    sequence_id: str = "sequence",
    allow_center_fallback: bool = True,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> MutationRecord:
    """Generate a paired corrupted sequence for a biological task.

    The edit is deterministic, length-preserving, and records exact coordinates.
    When a canonical motif is absent and `allow_center_fallback=True`, the
    function mutates a small center window so downstream smoke tests and controls
    still receive a valid paired sequence.
    """

    normalized_task = task.lower()
    if normalized_task in {"ctcf", "ctcf_gm12878", "ctcf_binding"}:
        task = "ctcf"
    else:
        task = canonicalize_task_name(task)
    clean = sequence.upper()

    if task == "promoter_tata":
        span = _first_regex_span(TATA_RE, clean)
        motif_name = "TATA-like"
        strategy = "regex_tata_to_gc_balanced"
        fallback_length = 6
    elif task == "splice_sites_donors":
        span = _find_near_center(clean, "GT")
        motif_name = "splice_donor_GT"
        strategy = "splice_donor_gt_to_gc"
        fallback_length = 2
    elif task == "splice_sites_acceptors":
        span = _find_near_center(clean, "AG")
        motif_name = "splice_acceptor_AG"
        strategy = "splice_acceptor_ag_to_ac"
        fallback_length = 2
    elif task == "promoter_no_tata":
        span = _fallback_center_span(clean, 6)
        motif_name = "promoter_no_tata_control"
        strategy = "promoter_no_tata_center_control"
        fallback_length = 6
    else:
        span = _find_ctcf_span(clean, config)
        motif_name = "CTCF_core"
        strategy = "ctcf_core_to_gc_balanced"
        fallback_length = 14

    if span is None:
        if not allow_center_fallback:
            raise ValueError(f"No mutable motif found for task '{task}' in sequence '{sequence_id}'.")
        span = _fallback_center_span(clean, fallback_length)
        strategy = f"{strategy}_center_fallback"

    start, end = span
    original = clean[start:end]
    if task == "splice_sites_donors" and len(original) == 2:
        replacement = "GC"
    elif task == "splice_sites_acceptors" and len(original) == 2:
        replacement = "AC"
    else:
        replacement = _gc_balanced_replacement(end - start)
    if replacement == original:
        replacement = ("CG" * ((len(original) // 2) + 1))[: len(original)]

    corrupted = _replace_span(clean, start, end, replacement)
    if len(corrupted) != len(clean):
        raise AssertionError("Counterfactual sequence length changed.")

    return MutationRecord(
        sequence_id=sequence_id,
        task=task,
        clean_sequence=clean,
        corrupted_sequence=corrupted,
        motif_name=motif_name,
        start=start,
        end=end,
        clean_subsequence=original,
        corrupted_subsequence=replacement,
        strategy=strategy,
    )


def generate_counterfactual_pairs(
    sequences: Iterable[str],
    task: str,
    sequence_ids: Iterable[str] | None = None,
) -> list[MutationRecord]:
    """Generate counterfactual records for a sequence collection."""

    sequence_list = list(sequences)
    ids = list(sequence_ids) if sequence_ids is not None else [f"sequence_{idx}" for idx in range(len(sequence_list))]
    if len(ids) != len(sequence_list):
        raise ValueError("sequence_ids must have the same length as sequences.")
    return [
        generate_counterfactual_sequence(sequence=sequence, task=task, sequence_id=str(sequence_id))
        for sequence, sequence_id in zip(sequence_list, ids)
    ]


def save_counterfactual_pairs(
    records: list[MutationRecord],
    output_name: str,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Path:
    """Save paired clean/corrupted sequences and a manifest under `results/`."""

    config.ensure_paths()
    output_path = config.paths.counterfactuals_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(record) for record in records]).to_csv(output_path, sep="\t", index=False)
    write_json(
        config.paths.manifests_dir / f"{Path(output_name).stem}_counterfactuals_manifest.json",
        {
            "created_at": utc_now_iso(),
            "path": str(output_path),
            "records": len(records),
            "tasks": sorted({record.task for record in records}),
        },
    )
    return output_path


def char_span_to_token_span(
    sequence: str,
    char_start: int,
    char_end: int,
    tokenizer,
) -> tuple[int, int]:
    """Map a character motif span to tokenizer positions using offsets when available."""

    encoded = tokenizer(
        sequence,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=False,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("Tokenizer did not return offset mappings; cannot map motif span exactly.")

    token_indices = [
        idx
        for idx, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end and end > start
    ]
    if not token_indices:
        raise ValueError(f"No tokenizer positions overlap character span [{char_start}, {char_end}).")
    return min(token_indices), max(token_indices) + 1
