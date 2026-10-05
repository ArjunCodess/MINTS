"""Run the fixed native masked-flank pilot on discovery chromosomes only."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
import platform
import subprocess
import time

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG
from src.controlled_edits import MotifDefinition, controlled_edit_pairs
from src.motif_scoring import load_jaspar_ctcf_motif, motif_pssm
from src.native_endpoint import NativeProtocol, load_native_mlm, prepare_target, native_score, summarize_pilot
from src.utils import sha256_file, write_json


ROOT = Path(__file__).resolve().parents[1]


def discovery_population(table, protocol):
    """Filter chromosomes before motif generation or model evaluation."""
    required = {"chrom", "start", "end", "sequence"}
    if not required.issubset(table.columns):
        raise ValueError(f"Input must contain {sorted(required)}")
    rows = table[table.chrom.isin(protocol.discovery_chromosomes)].copy()
    rows["sequence_id"] = rows.chrom + ":" + rows.start.astype(str) + "-" + rows.end.astype(str)
    if rows.sequence_id.duplicated().any():
        raise ValueError("Discovery coordinates must be unique")
    if any(rows.end - rows.start != rows.sequence.str.len()):
        raise ValueError("Discovery sequence lengths must match genomic intervals")
    rows = rows.sort_values("sequence_id").reset_index(drop=True)
    return rows.iloc[np.random.default_rng(protocol.seed).permutation(len(rows))[:protocol.scan_cap]]


def run(output, input_path, device="auto", protocol=None):
    protocol = protocol or NativeProtocol()
    output, input_path = Path(output).resolve(), Path(input_path).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory; completed pilots cannot be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "src/native_endpoint.py", Path(__file__).resolve(),
               ROOT / "src/modeling.py", ROOT / "src/controlled_edits.py",
               ROOT / "src/assay_alignment.py", ROOT / "src/assay_stats.py",
               ROOT / "src/motif_scoring.py", ROOT / "src/config.py", ROOT / "src/utils.py"]
    source_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in sources}
    # Save the entire endpoint and gate before loading the model or observing scores.
    write_json(output / "protocol.json", dict(**asdict(protocol),
        model_name=DEFAULT_CONFIG.model.model_name, model_revision=DEFAULT_CONFIG.model.revision,
        motif="JASPAR MA0139.1", motif_threshold_fraction=0.8, one_edit_per_sequence=True,
        cluster_block_bp=1_000_000, analysis_status="discovery only; not preregistered confirmation",
        eligibility="canonical DNA, matched transitions/composition, exact offsets, unchanged visible flank",
        confirmation_rule="no reuse of inspected chromosomes 20/21 or pilot sequences",
        controls=["transition/width/token-budget matched non-motif edit", "identity scores",
                  "whole-context rescue sensitivity before head-specific null interpretation"],
        source_sha256=source_hashes, input_sha256=sha256_file(input_path)))
    started = time.time()
    receipt = dict(status="running", started_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                   protocol_sha256=sha256_file(output / "protocol.json"), source_sha256=source_hashes,
                   command=sys.argv, python=platform.python_version(),
                   packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "pandas")},
                   git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    write_json(output / "execution.json", receipt)
    rows, eligibility, pairs = [], [], []
    try:
        population = discovery_population(pd.read_csv(input_path, sep="\t"), protocol)
        bundle, audit = load_native_mlm(replace(DEFAULT_CONFIG.model, device=device, local_files_only=True))
        write_json(output / "checkpoint.json", audit)
        motif = MotifDefinition("CTCF", pssm=motif_pssm(load_jaspar_ctcf_motif()), fraction=.8)
        for index, row in enumerate(population.itertuples()):
            candidates, entry = controlled_edit_pairs(row.sequence, row.sequence_id, motif, bundle.tokenizer,
                max_candidates=protocol.edit_candidates, max_pairs=1, seed=protocol.seed + index)
            eligibility.append(entry)
            if not candidates:
                continue
            pair = candidates[0]
            try:
                target = prepare_target(bundle.tokenizer, pair, protocol)
            except ValueError as exc:
                entry["target_exclusion"] = str(exc)
                continue
            scores = [native_score(bundle, ids, target) for ids in target["masked_ids"]]
            clean, edited, sham = [s["log_probability"] for s in scores]
            rows.append(dict(sequence_id=row.sequence_id, clean_sequence=row.sequence,
                target_index=target["index"], target_token_id=target["token_id"],
                target_start=target["span"][0], target_end=target["span"][1], target_nucleotides=target["nucleotides"],
                clean_log_probability=clean, edited_log_probability=edited, sham_log_probability=sham,
                clean_rank=scores[0]["rank"], motif_loss=clean-edited, sham_loss=clean-sham,
                motif_minus_sham_loss=sham-edited))
            pairs.append(dict(pair=pair, target=target))
            print(f"native pilot: {len(rows)}/{protocol.sequence_cap} retained", flush=True)
            if len(rows) >= protocol.sequence_cap:
                break
        write_json(output / "eligibility.json", dict(scanned=len(eligibility), entries=eligibility))
        with (output / "pairs.jsonl").open("w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair) + "\n")
        pd.DataFrame(rows, columns=["sequence_id", "clean_sequence", "target_index", "target_token_id",
            "target_start", "target_end", "target_nucleotides", "clean_log_probability", "edited_log_probability",
            "sham_log_probability", "clean_rank", "motif_loss", "sham_loss", "motif_minus_sham_loss"]
            ).to_csv(output / "scores.csv", index=False)
        summary = summarize_pilot(rows, protocol)
        summary["next_action"] = ("design discovery intervention and freeze fresh confirmation before execution"
            if summary["status"] == "eligible_for_discovery_intervention"
            else "stop this endpoint; any revised endpoint is a new exploratory study")
        write_json(output / "summary.json", summary)
        receipt["status"] = "completed"
        return summary
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["elapsed_seconds"] = time.time() - started
        receipt["source_changed_during_run"] = any(sha256_file(ROOT / n) != h for n, h in source_hashes.items())
        receipt["artifacts"] = {p.name: sha256_file(p) for p in sorted(output.iterdir())
                                if p.is_file() and p.name != "execution.json"}
        write_json(output / "execution.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/native_endpoint")
    parser.add_argument("--input", type=Path, default=DEFAULT_CONFIG.paths.ctcf_dir / "ctcf_gm12878_sequences.tsv")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.input, args.device), indent=2))
