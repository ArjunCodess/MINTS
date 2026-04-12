"""CTCF sequence preparation for ENCODE GM12878 inputs."""

from __future__ import annotations

import gzip
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_CONFIG, PipelineConfig
from .data_ingestion import download_file
from .utils import sha256_file, utc_now_iso, write_json


@dataclass(frozen=True)
class Interval:
    """A genomic interval parsed from a BED-like file."""

    chrom: str
    start: int
    end: int
    name: str
    score: float | None = None


def ensure_grch38_fasta(
    config: PipelineConfig = DEFAULT_CONFIG,
    overwrite: bool = False,
    decompress: bool = True,
) -> Path:
    """Download and optionally decompress the UCSC hg38 FASTA reference."""

    config.ensure_paths()
    record = download_file(
        config.data.grch38_fasta_url,
        config.paths.grch38_fasta_gz,
        overwrite=overwrite,
    )
    fasta_path = config.paths.grch38_fasta
    if decompress and (overwrite or not fasta_path.exists()):
        with gzip.open(config.paths.grch38_fasta_gz, "rb") as source:
            with fasta_path.open("wb") as target:
                shutil.copyfileobj(source, target)

    write_json(
        config.paths.manifests_dir / "grch38_manifest.json",
        {
            "created_at": utc_now_iso(),
            "url": config.data.grch38_fasta_url,
            "download": {
                "path": str(record.output_path),
                "status": record.status,
                "bytes": record.bytes,
                "sha256": record.sha256,
            },
            "fasta_path": str(fasta_path if fasta_path.exists() else config.paths.grch38_fasta_gz),
            "fasta_sha256": sha256_file(fasta_path) if fasta_path.exists() else None,
        },
    )
    return fasta_path if fasta_path.exists() else config.paths.grch38_fasta_gz


def _open_text_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def read_bed_intervals(path: Path, limit: int | None = None) -> list[Interval]:
    """Parse intervals from a BED or BED.GZ file."""

    intervals: list[Interval] = []
    with _open_text_maybe_gzip(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            name = fields[3] if len(fields) > 3 else f"{fields[0]}:{fields[1]}-{fields[2]}"
            score = None
            if len(fields) > 4:
                try:
                    score = float(fields[4])
                except ValueError:
                    score = None
            intervals.append(
                Interval(
                    chrom=fields[0],
                    start=int(fields[1]),
                    end=int(fields[2]),
                    name=name,
                    score=score,
                )
            )
            if limit is not None and len(intervals) >= limit:
                break
    return intervals


def extract_sequences_from_intervals(
    fasta_path: Path,
    intervals: list[Interval],
    output_tsv: Path,
    flank: int = 0,
) -> Path:
    """Extract DNA sequences for BED intervals into a TSV file."""

    from pyfaidx import Fasta

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    fasta = Fasta(str(fasta_path), rebuild=True)
    with output_tsv.open("w", encoding="utf-8") as handle:
        handle.write("name\tchrom\tstart\tend\tscore\tsequence\n")
        for interval in intervals:
            start = max(0, interval.start - flank)
            end = interval.end + flank
            sequence = str(fasta[interval.chrom][start:end]).upper()
            handle.write(
                f"{interval.name}\t{interval.chrom}\t{start}\t{end}\t"
                f"{'' if interval.score is None else interval.score}\t{sequence}\n"
            )
    return output_tsv


def prepare_ctcf_sequences(
    config: PipelineConfig = DEFAULT_CONFIG,
    bed_path: Path | None = None,
    fasta_path: Path | None = None,
    limit: int | None = None,
    flank: int = 0,
) -> Path:
    """Create a CTCF sequence TSV from an ENCODE BED and GRCh38 FASTA."""

    config.ensure_paths()
    if bed_path is None:
        bed_candidates = sorted(config.paths.encode_dir.glob("*.bed.gz")) + sorted(
            config.paths.encode_dir.glob("*.bed")
        )
        if not bed_candidates:
            raise FileNotFoundError(
                f"No BED/BED.GZ files found in {config.paths.encode_dir}. "
                "Run `python main.py download-encode` first."
            )
        bed_path = bed_candidates[0]
    if fasta_path is None:
        fasta_path = config.paths.grch38_fasta
    if not fasta_path.exists():
        raise FileNotFoundError(
            f"GRCh38 FASTA not found at {fasta_path}. Run `python main.py download-grch38` first."
        )

    intervals = read_bed_intervals(bed_path, limit=limit)
    output_tsv = config.paths.ctcf_dir / "ctcf_gm12878_sequences.tsv"
    extract_sequences_from_intervals(fasta_path, intervals, output_tsv, flank=flank)
    write_json(
        config.paths.manifests_dir / "ctcf_sequences_manifest.json",
        {
            "created_at": utc_now_iso(),
            "bed_path": str(bed_path),
            "fasta_path": str(fasta_path),
            "output_tsv": str(output_tsv),
            "intervals": len(intervals),
            "flank": flank,
        },
    )
    return output_tsv
