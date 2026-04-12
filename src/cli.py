"""Command-line entrypoint for the MINTS pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

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
    parser.add_argument(
        "--max-probe-train",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Limit residual caching and probe training to N train examples per task. "
            "Omit this flag to use all available train rows."
        ),
    )
    parser.add_argument(
        "--max-probe-test",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Limit residual caching and probe evaluation to N test examples per task. "
            "Omit this flag to use all available test rows."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    return parser


def run(argv: list[str] | None = None) -> int:
    """Run the complete pipeline."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_probe_train is not None and args.max_probe_train <= 0:
        parser.error("--max-probe-train must be a positive integer.")
    if args.max_probe_test is not None and args.max_probe_test <= 0:
        parser.error("--max-probe-test must be a positive integer.")

    config = PipelineConfig()
    config = replace(
        config,
        data=replace(
            config.data,
            max_probe_train=args.max_probe_train,
            max_probe_test=args.max_probe_test,
        ),
    )
    manifest_path = run_pipeline(config=config, overwrite=args.overwrite)
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
