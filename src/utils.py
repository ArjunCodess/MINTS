"""Shared utility helpers for MINTS."""

from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def require_package(import_name: str, install_name: str | None = None) -> Any:
    """Import a dependency or raise an actionable error."""

    try:
        return __import__(import_name)
    except ImportError as exc:
        package = install_name or import_name
        raise ImportError(
            f"Missing dependency '{package}'. Install project requirements with "
            "`python -m pip install -r requirements.txt` in Python 3.11/3.12."
        ) from exc


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA-256 checksum for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def set_reproducibility_seed(seed: int) -> None:
    """Seed common pseudo-random generators used by the pipeline."""

    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
