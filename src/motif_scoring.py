"""Ground-truth CTCF motif scoring for strict circuit tests.

This module keeps biological ground truth separate from model internals. It
loads a JASPAR CTCF PWM, scans nucleotide sequences, and maps the resulting
motif evidence onto tokenizer positions so QK scores and attention maps can be
compared against the same support intervals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .enrichment import MotifSupport
from .utils import progress, utc_now_iso, write_json


JASPAR_CTCF_ID = "MA0139.1"
JASPAR_2024_CTCF_URL = "https://jaspar2024.elixir.no/api/v1/matrix/MA0139.1.jaspar"


@dataclass(frozen=True)
class TokenMotifScores:
    """Motif evidence projected from nucleotide positions to token positions."""

    sequence_index: int
    sequence_id: str
    sequence: str
    token_scores: np.ndarray
    token_offsets: list[tuple[int, int]]
    support_tokens: list[int]
    threshold: float


def find_jaspar_matrix_path(
    config: PipelineConfig = DEFAULT_CONFIG,
    matrix_id: str = JASPAR_CTCF_ID,
) -> Path:
    """Find a local JASPAR matrix file for the requested matrix ID."""

    candidates = sorted(config.paths.data_dir.glob(f"**/{matrix_id}.jaspar"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {matrix_id}.jaspar under {config.paths.data_dir}. "
            "Place the JASPAR CORE vertebrate CTCF matrix in data/ before running strict motif tests."
        )
    return candidates[0]


def ensure_jaspar_ctcf_matrix(
    config: PipelineConfig = DEFAULT_CONFIG,
    matrix_id: str = JASPAR_CTCF_ID,
    url: str = JASPAR_2024_CTCF_URL,
) -> Path:
    """Return the local CTCF matrix path, downloading the small JASPAR file if needed."""

    try:
        return find_jaspar_matrix_path(config=config, matrix_id=matrix_id)
    except FileNotFoundError:
        pass

    import requests

    output_dir = config.paths.data_dir / "jaspar_2024_core_vertebrates"
    output_path = output_dir / f"{matrix_id}.jaspar"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress(f"Downloading JASPAR CTCF matrix {matrix_id} from {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    output_path.write_text(response.text, encoding="utf-8")
    return output_path


def load_jaspar_ctcf_motif(
    path: Path | None = None,
    config: PipelineConfig = DEFAULT_CONFIG,
):
    """Load the JASPAR CTCF motif using Biopython's JASPAR parser."""

    from Bio import motifs

    matrix_path = path or ensure_jaspar_ctcf_matrix(config)
    with matrix_path.open("r", encoding="utf-8") as handle:
        motif = motifs.read(handle, "jaspar")
    if not str(motif.matrix_id).startswith(JASPAR_CTCF_ID):
        raise ValueError(f"Expected CTCF matrix {JASPAR_CTCF_ID}, got {motif.matrix_id}.")
    return motif


def motif_pssm(motif: Any, pseudocounts: float = 0.5):
    """Return a log-odds PSSM from a Biopython motif object."""

    pwm = motif.counts.normalize(pseudocounts=pseudocounts)
    return pwm.log_odds()


def scan_sequence_with_pssm(sequence: str, pssm: Any, include_reverse_complement: bool = True) -> np.ndarray:
    """Scan a DNA sequence and return one motif-start score per nucleotide index.

    Scores are defined at motif start positions. Positions that cannot start a
    full motif receive `-inf`, so downstream token aggregation cannot
    accidentally treat invalid windows as background support.
    """

    clean_sequence = sequence.upper()
    motif_length = int(pssm.length)
    scores = np.full(len(clean_sequence), -np.inf, dtype=np.float64)
    if len(clean_sequence) < motif_length:
        return scores

    forward = np.asarray(pssm.calculate(clean_sequence), dtype=np.float64)
    if include_reverse_complement:
        reverse = np.asarray(pssm.reverse_complement().calculate(clean_sequence), dtype=np.float64)
        start_scores = np.maximum(forward, reverse)
    else:
        start_scores = forward
    scores[: len(start_scores)] = start_scores
    return scores


def default_support_threshold(pssm: Any, fraction_of_max: float = 0.80) -> float:
    """Set a deterministic motif-support threshold from the PSSM score range."""

    min_score = float(pssm.min)
    max_score = float(pssm.max)
    return min_score + fraction_of_max * (max_score - min_score)


def token_offsets_for_sequence(tokenizer: Any, sequence: str) -> list[tuple[int, int]]:
    """Return tokenizer character offsets in model-token coordinates.

    Special tokens usually have zero-width offsets. They are retained and later
    assigned NaN motif scores so biological evidence stays aligned to hidden
    states, attention maps, and QK scores that include special tokens.
    """

    encoded = tokenizer(
        sequence,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=False,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("Tokenizer did not return offset mappings; exact motif-to-token mapping is unavailable.")
    return [(int(start), int(end)) for start, end in offsets]


def aggregate_start_scores_to_tokens(
    start_scores: np.ndarray,
    motif_length: int,
    token_offsets: list[tuple[int, int]],
) -> np.ndarray:
    """Map motif-start scores to token positions by overlap with motif windows."""

    token_scores = np.full(len(token_offsets), np.nan, dtype=np.float64)
    finite_starts = np.flatnonzero(np.isfinite(start_scores))
    for token_idx, (token_start, token_end) in enumerate(token_offsets):
        if token_end <= token_start:
            continue
        overlapping_scores = [
            start_scores[start]
            for start in finite_starts
            if start < token_end and (start + motif_length) > token_start
        ]
        token_scores[token_idx] = float(np.max(overlapping_scores)) if overlapping_scores else np.nan
    return token_scores


def score_sequence_tokens(
    sequence: str,
    tokenizer: Any,
    motif: Any,
    sequence_index: int = 0,
    sequence_id: str = "sequence",
    threshold: float | None = None,
) -> TokenMotifScores:
    """Score every model token position for CTCF motif support."""

    pssm = motif_pssm(motif)
    support_threshold = default_support_threshold(pssm) if threshold is None else threshold
    start_scores = scan_sequence_with_pssm(sequence, pssm)
    offsets = token_offsets_for_sequence(tokenizer, sequence)
    token_scores = aggregate_start_scores_to_tokens(start_scores, int(pssm.length), offsets)
    support_tokens = [
        token_idx for token_idx, score in enumerate(token_scores) if np.isfinite(score) and score >= support_threshold
    ]
    return TokenMotifScores(
        sequence_index=sequence_index,
        sequence_id=sequence_id,
        sequence=sequence.upper(),
        token_scores=token_scores,
        token_offsets=offsets,
        support_tokens=support_tokens,
        threshold=float(support_threshold),
    )


def motif_supports_from_scores(scores: Iterable[TokenMotifScores], motif_name: str = "CTCF") -> list[MotifSupport]:
    """Convert token-level motif evidence into one-token support records."""

    supports: list[MotifSupport] = []
    for record in scores:
        for token_idx in record.support_tokens:
            supports.append(
                MotifSupport(
                    sequence_index=record.sequence_index,
                    token_start=token_idx,
                    token_end=token_idx + 1,
                    motif_name=motif_name,
                )
            )
    return supports


def load_ctcf_sequence_table(config: PipelineConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Read the prepared ENCODE GM12878 CTCF sequence table."""

    path = config.paths.ctcf_dir / "ctcf_gm12878_sequences.tsv"
    if not path.exists():
        raise FileNotFoundError(f"CTCF sequence table not found: {path}")
    table = pd.read_csv(path, sep="\t")
    if "sequence" not in table.columns:
        raise ValueError(f"CTCF sequence table must contain a 'sequence' column: {path}")
    return table


def score_ctcf_table(
    tokenizer: Any,
    config: PipelineConfig = DEFAULT_CONFIG,
    max_sequences: int | None = None,
    threshold: float | None = None,
) -> list[TokenMotifScores]:
    """Score CTCF motif support for prepared ENCODE GM12878 sequences."""

    motif = load_jaspar_ctcf_motif(config=config)
    table = load_ctcf_sequence_table(config)
    if max_sequences is not None:
        table = table.head(max_sequences)
    records: list[TokenMotifScores] = []
    for sequence_index, row in enumerate(table.itertuples(index=False)):
        sequence = str(getattr(row, "sequence"))
        sequence_id = str(getattr(row, "name", f"ctcf_{sequence_index}"))
        records.append(
            score_sequence_tokens(
                sequence=sequence,
                tokenizer=tokenizer,
                motif=motif,
                sequence_index=sequence_index,
                sequence_id=sequence_id,
                threshold=threshold,
            )
        )
    return records


def save_token_motif_scores(
    records: list[TokenMotifScores],
    output_name: str,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Path:
    """Save token-level motif scores under `results/enrichment/`."""

    config.ensure_paths()
    output_path = config.paths.enrichment_dir / output_name
    rows: list[dict[str, Any]] = []
    for record in records:
        support_tokens = set(record.support_tokens)
        for token_idx, score in enumerate(record.token_scores):
            start, end = record.token_offsets[token_idx]
            rows.append(
                {
                    "sequence_index": record.sequence_index,
                    "sequence_id": record.sequence_id,
                    "token": token_idx,
                    "char_start": start,
                    "char_end": end,
                    "motif_score": float(score) if np.isfinite(score) else np.nan,
                    "is_support": token_idx in support_tokens,
                    "threshold": record.threshold,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    write_json(
        config.paths.manifests_dir / f"{Path(output_name).stem}_motif_scores_manifest.json",
        {
            "created_at": utc_now_iso(),
            "path": str(output_path),
            "records": len(records),
            "matrix_id": JASPAR_CTCF_ID,
            "jaspar_url": JASPAR_2024_CTCF_URL,
        },
    )
    progress(f"Wrote token motif scores: {output_path}")
    return output_path
