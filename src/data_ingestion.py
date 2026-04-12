"""Data ingestion utilities for the MINTS pipeline."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

from .config import DEFAULT_CONFIG, DataConfig, PipelineConfig
from .utils import progress, sha256_file, utc_now_iso, write_json


ENCODE_DOWNLOAD_RE = re.compile(r"^https://www\.encodeproject\.org/files/[^/]+/@@download/[^\"']+$")


@dataclass(frozen=True)
class DownloadRecord:
    """Metadata for a downloaded or discovered artifact."""

    url: str
    output_path: Path
    status: str
    bytes: int | None = None
    sha256: str | None = None


def canonicalize_task_name(task_name: str, data_config: DataConfig = DEFAULT_CONFIG.data) -> str:
    """Map accepted aliases to canonical HF task names."""

    task = data_config.canonical_task(task_name)
    if task not in data_config.task_names:
        allowed = ", ".join(sorted(set(data_config.task_names) | set(data_config.task_aliases)))
        raise ValueError(f"Unknown task '{task_name}'. Expected one of: {allowed}")
    return task


def read_encode_urls(path: Path = DEFAULT_CONFIG.paths.encode_url_file) -> list[str]:
    """Read URL-like lines from the ENCODE input text file."""

    if not path.exists():
        raise FileNotFoundError(f"ENCODE URL file does not exist: {path}")

    urls: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().strip('"').strip("'")
        if not line or line.startswith("#"):
            continue
        if line.startswith("http://") or line.startswith("https://"):
            urls.append(line)
    return urls


def is_encode_artifact_url(url: str, data_config: DataConfig = DEFAULT_CONFIG.data) -> bool:
    """Return True for direct ENCODE file downloads we want to materialize."""

    if not ENCODE_DOWNLOAD_RE.match(url):
        return False
    parsed_path = unquote(urlparse(url).path)
    return parsed_path.endswith(data_config.encode_allowed_suffixes)


def encode_output_filename(url: str) -> str:
    """Derive a deterministic local filename from an ENCODE download URL."""

    filename = Path(unquote(urlparse(url).path)).name
    if not filename:
        raise ValueError(f"Could not derive filename from URL: {url}")
    return filename


def filter_encode_artifact_urls(urls: Iterable[str]) -> list[str]:
    """Filter URL lines to the direct artifact downloads used by the pipeline."""

    return [url for url in urls if is_encode_artifact_url(url)]


def download_file(url: str, output_path: Path, overwrite: bool = False) -> DownloadRecord:
    """Download a URL to disk with streaming writes and checksum metadata."""

    import requests
    from tqdm.auto import tqdm

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return DownloadRecord(
            url=url,
            output_path=output_path,
            status="exists",
            bytes=output_path.stat().st_size,
            sha256=sha256_file(output_path),
        )

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        with tmp_path.open("wb") as handle:
            progress = tqdm(total=total, unit="B", unit_scale=True, desc=output_path.name)
            try:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        progress.update(len(chunk))
            finally:
                progress.close()
    tmp_path.replace(output_path)
    return DownloadRecord(
        url=url,
        output_path=output_path,
        status="downloaded",
        bytes=output_path.stat().st_size,
        sha256=sha256_file(output_path),
    )


def download_encode_artifacts(
    config: PipelineConfig = DEFAULT_CONFIG,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[DownloadRecord]:
    """Download CTCF GM12878 ENCODE artifacts listed in the project data file."""

    config.ensure_paths()
    all_urls = read_encode_urls(config.paths.encode_url_file)
    artifact_urls = filter_encode_artifact_urls(all_urls)
    records: list[DownloadRecord] = []

    for url in artifact_urls:
        output_path = config.paths.encode_dir / encode_output_filename(url)
        progress(f"Checking ENCODE artifact: {output_path.name}")
        if dry_run:
            records.append(DownloadRecord(url=url, output_path=output_path, status="dry_run"))
        else:
            records.append(download_file(url, output_path, overwrite=overwrite))

    write_json(
        config.paths.manifests_dir / "encode_ctcf_gm12878_manifest.json",
        {
            "created_at": utc_now_iso(),
            "url_file": str(config.paths.encode_url_file),
            "skipped_urls": [url for url in all_urls if url not in artifact_urls],
            "records": [
                {
                    "url": record.url,
                    "output_path": str(record.output_path),
                    "status": record.status,
                    "bytes": record.bytes,
                    "sha256": record.sha256,
                }
                for record in records
            ],
        },
    )
    return records


def load_hf_downstream_dataset(config: PipelineConfig = DEFAULT_CONFIG):
    """Load the revised nucleotide-transformer downstream dataset."""

    from datasets import load_dataset

    return load_dataset(config.data.hf_dataset_name, config.data.hf_dataset_config)


def _tokenize_dataset(dataset, tokenizer, max_length: int | None):
    """Tokenize the `sequence` column while preserving labels and metadata."""

    def tokenize_batch(batch):
        kwargs = {"truncation": True, "padding": False}
        if max_length is not None:
            kwargs["max_length"] = max_length
        return tokenizer(batch["sequence"], **kwargs)

    return dataset.map(tokenize_batch, batched=True, desc="Tokenizing sequences")


def _load_existing_dataset_dict(output_dir: Path):
    """Load a saved DatasetDict, returning None for partial/corrupt outputs."""

    from datasets import DatasetDict, load_from_disk

    if not output_dir.exists():
        return None
    try:
        existing = load_from_disk(str(output_dir))
    except (FileNotFoundError, ValueError, OSError):
        return None
    if not isinstance(existing, DatasetDict):
        return None
    return existing


def ingest_hf_downstream_tasks(
    config: PipelineConfig = DEFAULT_CONFIG,
    task_names: Iterable[str] | None = None,
    limit_per_split: int | None = None,
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    """Download, filter, tokenize, and save selected HF downstream tasks."""

    from datasets import DatasetDict, load_from_disk
    from transformers import AutoTokenizer

    config.ensure_paths()
    selected_tasks = tuple(task_names or config.data.task_names)
    canonical_tasks = tuple(canonicalize_task_name(task, config.data) for task in selected_tasks)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        trust_remote_code=config.model.trust_remote_code,
        revision=config.model.revision,
    )
    dataset_dict = load_hf_downstream_dataset(config)
    summary: dict[str, dict[str, int]] = {}

    for task in canonical_tasks:
        progress(f"Preparing HF downstream task: {task}")
        task_splits = {}
        for split_name, split_dataset in dataset_dict.items():
            progress(f"Filtering {split_name} split for task={task}")
            filtered = split_dataset.filter(
                lambda example, task_name=task: example["task"] == task_name,
                desc=f"Filtering {split_name}:{task}",
            )
            if limit_per_split is not None:
                filtered = filtered.select(range(min(limit_per_split, len(filtered))))
            task_splits[split_name] = filtered

        task_dataset = DatasetDict(task_splits)
        if sum(len(split) for split in task_dataset.values()) == 0:
            raise ValueError(f"No rows found for task '{task}' in {config.data.hf_dataset_name}.")

        tokenized = _tokenize_dataset(task_dataset, tokenizer, config.data.token_max_length)
        output_dir = config.paths.hf_downstream_dir / task
        if output_dir.exists() and overwrite:
            shutil.rmtree(output_dir)
        if output_dir.exists() and not overwrite:
            existing = _load_existing_dataset_dict(output_dir)
            if existing is not None:
                summary[task] = {split: len(ds) for split, ds in existing.items()}
                progress(f"Reusing cached tokenized dataset for {task}: {summary[task]}")
                continue
            progress(f"Removing partial cached dataset for {task}: {output_dir}")
            shutil.rmtree(output_dir)
        progress(f"Saving tokenized dataset for {task} to {output_dir}")
        tokenized.save_to_disk(str(output_dir))
        summary[task] = {split: len(ds) for split, ds in tokenized.items()}

    write_json(
        config.paths.manifests_dir / "hf_downstream_manifest.json",
        {
            "created_at": utc_now_iso(),
            "dataset": config.data.hf_dataset_name,
            "dataset_config": config.data.hf_dataset_config,
            "model_tokenizer": config.model.model_name,
            "tasks": summary,
            "limit_per_split": limit_per_split,
        },
    )
    return summary
