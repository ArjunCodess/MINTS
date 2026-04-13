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
    activations_dir: Path = RESULTS_DIR / "activations"
    circuits_dir: Path = RESULTS_DIR / "circuits"
    enrichment_dir: Path = RESULTS_DIR / "enrichment"
    qk_alignment_dir: Path = RESULTS_DIR / "qk_alignment"
    counterfactuals_dir: Path = RESULTS_DIR / "counterfactuals"
    patching_dir: Path = RESULTS_DIR / "patching"
    distributed_features_dir: Path = RESULTS_DIR / "distributed_features"
    cross_model_dir: Path = RESULTS_DIR / "cross_model"
    figures_dir: Path = RESULTS_DIR / "figures"
    tables_dir: Path = RESULTS_DIR / "tables"
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
            self.activations_dir,
            self.circuits_dir,
            self.enrichment_dir,
            self.qk_alignment_dir,
            self.counterfactuals_dir,
            self.patching_dir,
            self.distributed_features_dir,
            self.cross_model_dir,
            self.figures_dir,
            self.tables_dir,
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
    activation_layers: tuple[int, ...] = (0, 5, 11)
    probe_layer: int = 11
    max_probe_train: int | None = None
    max_probe_test: int | None = None
    max_qk_alignment_sequences: int | None = None
    max_patching_pairs: int = 500
    max_feature_search_sequences: int | None = 2048
    max_cross_model_qk_alignment_sequences: int | None = None
    sae_dictionary_size: int = 512
    sae_epochs: int = 10
    sae_batch_size: int = 256
    sae_learning_rate: float = 1e-3
    sae_l1_coefficient: float = 1e-3
    circuit_layers: tuple[int, ...] = (0, 5, 11)

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
