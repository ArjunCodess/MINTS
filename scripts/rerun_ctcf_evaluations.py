"""Rerun corrected CTCF mechanistic evaluations without the full pipeline.

This entrypoint is intentionally narrow. It reloads the configured nucleotide
transformer, optionally refreshes the QK/OV circuit archive, then reruns:

1. BPE-correct CTCF motif scoring, QK Pearson alignment, and matched attention
   enrichment.
2. Residual-vs-MLP distributed feature search using the corrected feed-forward
   post-activation hook target.

Example:
    python scripts/rerun_ctcf_evaluations.py --max-qk-sequences 512 --max-feature-sequences 512 --sae-epochs 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.circuits import extract_qk_ov_matrices  # noqa: E402
from src.config import DataConfig, PipelineConfig  # noqa: E402
from src.distributed_features import run_distributed_feature_search  # noqa: E402
from src.modeling import load_hooked_encoder  # noqa: E402
from src.qk_alignment import run_ctcf_qk_alignment  # noqa: E402
from src.utils import progress, utc_now_iso, write_json  # noqa: E402


def _positive_or_none(value: str) -> int | None:
    """Parse sequence-count CLI arguments, using `0` for the full dataset."""

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be non-negative.")
    return None if parsed == 0 else parsed


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the corrected CTCF evaluation rerun."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-qk-sequences",
        type=_positive_or_none,
        default=None,
        help="Maximum CTCF sequences for QK/enrichment scan. Use 0 for all. Default: all.",
    )
    parser.add_argument(
        "--max-feature-sequences",
        type=_positive_or_none,
        default=2048,
        help="Maximum CTCF sequences for SAE feature search. Use 0 for all. Default: 2048.",
    )
    parser.add_argument(
        "--sae-epochs",
        type=int,
        default=None,
        help="Override SAE training epochs for this rerun.",
    )
    parser.add_argument(
        "--skip-qk",
        action="store_true",
        help="Skip CTCF QK alignment and matched attention enrichment.",
    )
    parser.add_argument(
        "--skip-sae",
        action="store_true",
        help="Skip residual-vs-MLP SAE feature search.",
    )
    parser.add_argument(
        "--overwrite-circuits",
        action="store_true",
        help="Regenerate the all-layer QK/OV archive before the CTCF QK scan.",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    """Execute the corrected CTCF rerun and return the summary path."""

    data_config = DataConfig(
        max_qk_alignment_sequences=args.max_qk_sequences,
        max_feature_search_sequences=args.max_feature_sequences,
        sae_epochs=args.sae_epochs if args.sae_epochs is not None else DataConfig().sae_epochs,
    )
    config = PipelineConfig(data=data_config)
    config.ensure_paths()

    progress("Loading model for corrected CTCF evaluations")
    bundle = load_hooked_encoder(config.model)
    progress(f"Model ready on {bundle.device} using {bundle.instrumentation_backend}")

    outputs: dict[str, Any] = {"created_at": utc_now_iso()}
    qk_archive_path = config.paths.circuits_dir / "qk_ov_matrices.npz"
    if not args.skip_qk and (args.overwrite_circuits or not qk_archive_path.exists()):
        progress("Refreshing all-layer QK/OV archive before corrected CTCF scan")
        circuit_export = extract_qk_ov_matrices(bundle=bundle, config=config)
        outputs["circuits"] = {
            "path": str(circuit_export.path),
            "layers": list(circuit_export.layers),
            "n_heads": circuit_export.n_heads,
            "d_model": circuit_export.d_model,
            "d_head": circuit_export.d_head,
        }

    if not args.skip_qk:
        progress("Rerunning BPE-correct CTCF QK alignment and matched enrichment")
        qk_outputs = run_ctcf_qk_alignment(
            bundle=bundle,
            config=config,
            max_sequences=args.max_qk_sequences,
            output_stem="ctcf_bpe_corrected_qk_alignment",
        )
        outputs["ctcf_qk_alignment"] = {key: str(value) for key, value in qk_outputs.items()}

    if not args.skip_sae:
        progress("Rerunning corrected residual-vs-MLP SAE feature search")
        sae_outputs = run_distributed_feature_search(
            bundle=bundle,
            config=config,
            max_sequences=args.max_feature_sequences,
        )
        outputs["distributed_feature_search"] = _stringify_paths(sae_outputs)

    summary_path = config.paths.tables_dir / "ctcf_corrected_evaluation_rerun_summary.json"
    write_json(summary_path, outputs)
    progress(f"Wrote corrected CTCF rerun summary: {summary_path}")
    return summary_path


def _stringify_paths(value: Any) -> Any:
    """Recursively convert Path objects to strings for JSON output."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_paths(item) for item in value]
    return value


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    summary_path = run(args)
    print(f"Corrected CTCF evaluation rerun complete. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
