"""Build the native-pilot manuscript insert from verified saved evidence."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json

from src.utils import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def build(output=ROOT / "results/native_endpoint"):
    output = Path(output)
    receipt = json.loads((output / "execution.json").read_text())
    if receipt["status"] != "completed" or receipt["source_changed_during_run"]:
        raise ValueError("Native pilot execution is incomplete or changed during execution")
    for name, digest in receipt["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"Native pilot source changed: {name}")
    for name, digest in receipt["artifacts"].items():
        if sha256_file(output / name) != digest:
            raise ValueError(f"Native pilot artifact changed: {name}")
    summary = json.loads((output / "summary.json").read_text())
    checkpoint = json.loads((output / "checkpoint.json").read_text())
    for key in ("missing_keys", "mismatched_keys", "error_msgs"):
        if checkpoint["loading_info"][key]:
            raise ValueError("Native checkpoint has untrained or incompatible weights")
    if summary["status"] != "stop":
        raise ValueError("The current manuscript insert describes only the stopped pilot")
    text = (
        "\\paragraph{Discovery-only native masked-flank pilot.}\n"
        "A separate fixed-endpoint pilot loads the pinned pretrained masked-language-model head without fitting a readout. "
        "No prediction-head parameters are missing or newly initialized. The target is the nearest unchanged whole downstream "
        "flank token at a gap of 1--30 bases, with upstream fallback. Its nucleotide span and token identity agree across "
        "clean, composition-preserving motif-edited and transition-matched sham sequences, and masking leaves the motif edit visible. "
        f"On chromosomes 18/19, {summary['retained_sequences']} retained sequences form {summary['genomic_clusters']} transitive "
        f"genomic clusters. The mean motif-minus-sham target log-probability loss is ${summary['mean']:.4f}$ nats "
        f"with a 95\\% cluster interval $[{summary['ci_low']:.4f}, {summary['ci_high']:.4f}]$. "
        "This fails the prespecified feasibility requirement of at least eight clusters, a positive interval lower bound "
        "and a mean of at least 0.05 nats. We stop before head selection or confirmation, rather than search other outputs. "
        "The interval is too wide to establish a precise null. This result concerns the chosen sequence-prediction endpoint; "
        "it does not establish absence of a CTCF mechanism. The already inspected chromosomes 20/21 cannot supply fresh "
        "confirmation for any subsequent extension.\n"
    )
    target = ROOT / "paper/native_endpoint.tex"
    target.write_text(text, encoding="utf-8", newline="\n")
    write_json(output / "paper_manifest.json", dict(
        generator_sha256=sha256_file(Path(__file__)),
        execution_sha256=sha256_file(output / "execution.json"),
        summary_sha256=sha256_file(output / "summary.json"),
        tex_sha256=sha256_file(target),
        claim="failed discovery feasibility gate; no native head or binding inference"))
    print("verified native pilot and generated paper/native_endpoint.tex")


if __name__ == "__main__":
    build()
