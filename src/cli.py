"""Command-line entrypoint for the MINTS pipeline."""

from __future__ import annotations

import argparse
import json

from .config import PipelineConfig
from .reproduce import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the single-command pipeline parser."""

    parser = argparse.ArgumentParser(
        prog="mints",
        description="Run the complete MINTS reproducibility pipeline.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated datasets and downloaded artifacts.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    return parser


def run(argv: list[str] | None = None) -> int:
    """Run the complete pipeline."""

    parser = build_parser()
    args = parser.parse_args(argv)
    manifest_path = run_pipeline(config=PipelineConfig(), overwrite=args.overwrite)
    payload = {
        "message": f"Completed MINTS reproducibility run. Manifest: {manifest_path}",
        "manifest": str(manifest_path),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(payload["message"])
    return 0


def main() -> int:
    """Console-script compatible main function."""

    return run()
