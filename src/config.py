"""Central configuration for the MINTS pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout used by all pipeline stages."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR
    hf_downstream_dir: Path = DATA_DIR / "hf_downstream"
    encode_dir: Path = DATA_DIR / "encode" / "ctcf_gm12878"
    ctcf_dir: Path = DATA_DIR / "ctcf"
    manifests_dir: Path = RESULTS_DIR / "manifests"
    encode_url_file: Path = DATA_DIR / "ENCODE4_v1.5.1_GRCh38.txt"
    grch38_fasta_gz: Path = DATA_DIR / "genomes" / "hg38.fa.gz"
    grch38_fasta: Path = DATA_DIR / "genomes" / "hg38.fa"

    def ensure(self) -> None:
        """Create directories used by the pipeline."""

        for path in (
            self.data_dir,
            self.results_dir,
            self.hf_downstream_dir,
            self.encode_dir,
            self.ctcf_dir,
            self.manifests_dir,
            self.grch38_fasta.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    """Model and tokenizer defaults for the mechanistic pipeline."""

    model_name: str = "zhihan1996/DNABERT-2-117M"
    trust_remote_code: bool = True
    device: str = "auto"
    revision: str | None = None


@dataclass(frozen=True)
class DataConfig:
    """Dataset and download defaults."""

    hf_dataset_name: str = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
    hf_dataset_config: str = "default"
    task_names: tuple[str, ...] = (
        "promoter_tata",
        "promoter_no_tata",
        "splice_sites_donors",
        "splice_sites_acceptors",
    )
    task_aliases: dict[str, str] = field(
        default_factory=lambda: {
            "splice_sites_donor": "splice_sites_donors",
            "splice_sites_acceptor": "splice_sites_acceptors",
        }
    )
    encode_allowed_suffixes: tuple[str, ...] = (".bigWig", ".bed.gz", ".bigBed")
    grch38_fasta_url: str = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
    seed: int = 1729
    token_max_length: int | None = None
    batch_size: int = 8

    def canonical_task(self, task_name: str) -> str:
        """Return the canonical task identifier used by the HF dataset."""

        return self.task_aliases.get(task_name, task_name)


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration object for the pipeline."""

    paths: ProjectPaths = field(default_factory=ProjectPaths)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def ensure_paths(self) -> None:
        """Create required pipeline directories."""

        self.paths.ensure()

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to JSON-compatible primitives."""

        def normalize(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [normalize(item) for item in value]
            if isinstance(value, dict):
                return {str(key): normalize(item) for key, item in value.items()}
            if isinstance(value, list):
                return [normalize(item) for item in value]
            return value

        return normalize(asdict(self))


DEFAULT_CONFIG = PipelineConfig()
