"""Command-line entrypoint for the MINTS pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .config import PipelineConfig
from .reproduce import PIPELINE_STEPS, run_pipeline


FROM_STEP_ALIASES = {
    "all": "write_config",
    "strict_qk_proof_inputs": "strict_mechanistic_proofs",
    "strict_qk_proof": "strict_mechanistic_proofs",
    "strict_proofs": "strict_mechanistic_proofs",
    "batch_patching": "systematic_causal_intervention",
    "systematic_patching": "systematic_causal_intervention",
    "feature_search": "distributed_feature_search",
    "distributed_features": "distributed_feature_search",
}


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
    parser.add_argument(
        "--max-qk-alignment-sequences",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Limit CTCF sequences used for strict QK motif-alignment exports. "
            "Omit this flag to scan all prepared CTCF sequences."
        ),
    )
    parser.add_argument(
        "--max-patching-pairs",
        type=int,
        default=500,
        metavar="N",
        help=(
            "Limit systematic denoising activation-patching pairs per configured task. "
            "Defaults to 500."
        ),
    )
    parser.add_argument(
        "--max-feature-search-sequences",
        type=int,
        default=2048,
        metavar="N",
        help=(
            "Limit CTCF sequences used for residual/MLP sparse feature search. "
            "Defaults to 2048. Use 0 to scan all prepared CTCF sequences."
        ),
    )
    parser.add_argument(
        "--sae-epochs",
        type=int,
        default=10,
        metavar="N",
        help="Sparse autoencoder training epochs for distributed feature search. Defaults to 10.",
    )
    parser.add_argument(
        "--from-step",
        choices=(
            "all",
            *PIPELINE_STEPS,
            "strict_qk_proof_inputs",
            "strict_qk_proof",
            "strict_proofs",
            "batch_patching",
            "systematic_patching",
            "feature_search",
            "distributed_features",
        ),
        default="all",
        help=(
            "Start the pipeline from a named checkpoint and continue forward. "
            "Use systematic_causal_intervention to resume at the batch denoising patching work."
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
    if args.max_qk_alignment_sequences is not None and args.max_qk_alignment_sequences <= 0:
        parser.error("--max-qk-alignment-sequences must be a positive integer.")
    if args.max_patching_pairs is not None and args.max_patching_pairs <= 0:
        parser.error("--max-patching-pairs must be a positive integer.")
    if args.max_feature_search_sequences is not None and args.max_feature_search_sequences < 0:
        parser.error("--max-feature-search-sequences must be zero or a positive integer.")
    if args.sae_epochs is not None and args.sae_epochs <= 0:
        parser.error("--sae-epochs must be a positive integer.")

    config = PipelineConfig()
    from_step = FROM_STEP_ALIASES.get(args.from_step, args.from_step)
    max_feature_search_sequences = (
        None if args.max_feature_search_sequences == 0 else args.max_feature_search_sequences
    )
    config = replace(
        config,
        data=replace(
            config.data,
            max_probe_train=args.max_probe_train,
            max_probe_test=args.max_probe_test,
            max_qk_alignment_sequences=args.max_qk_alignment_sequences,
            max_patching_pairs=args.max_patching_pairs,
            max_feature_search_sequences=max_feature_search_sequences,
            sae_epochs=args.sae_epochs,
        ),
    )
    manifest_path = run_pipeline(config=config, overwrite=args.overwrite, from_step=from_step)
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
