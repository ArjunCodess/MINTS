"""Reproduce the review audit without modifying submitted result artifacts.

Run: .venv/Scripts/python.exe tools/review_audit.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import hashlib
import json
import platform
import time
import importlib.metadata
from collections import Counter
import numpy as np
import pandas as pd
from datasets import load_from_disk
from src.config import DEFAULT_CONFIG
from src.integrity import audit_splits, sequence_hash, near_duplicate_pairs
from src.probing import _matched_gc_indices
from src.inference import validate_metric

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/review"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def count_stages(table):
    support = table["is_support"].astype(str).str.lower().isin(["true", "1"])
    finite = np.isfinite(pd.to_numeric(table["motif_score"], errors="coerce"))
    thresholded = table["motif_score"] >= table["threshold"]
    return dict(token_rows=len(table), finite_token_rows=int(finite.sum()),
                unique_motif_support_tokens=int(support.sum()),
                finite_unique_motif_support_tokens=int((support & finite).sum()),
                thresholded_unique_motif_support_tokens=int((support & finite & thresholded).sum()),
                motif_hit_token_occurrences=int(table["support_span_count"].sum()),
                sequences=int(table["sequence_index"].nunique()),
                motif_absent_sequences=int(table.groupby("sequence_index")["is_support"].apply(
                    lambda v: ~v.astype(str).str.lower().isin(["true", "1"]).any()).sum()))


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(seed=1729, submission_commit="7112ec22b770c1651f24d33ac5f45fda3e986324",
                  python=platform.python_version(), versions={}, inputs={}, splits={}, gc_matching={}, patching={})
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "transformers", "datasets"]:
        report["versions"][name] = importlib.metadata.version(name)
    for task in DEFAULT_CONFIG.data.task_names:
        dataset = load_from_disk(str(ROOT / "data/hf_downstream" / task))
        parts = {split: list(rows) for split, rows in dataset.items()}
        audit = audit_splits(parts)
        audit["near_duplicates"] = near_duplicate_pairs(parts)
        audit["near_duplicate_status"] = "exhaustive <=2 substitutions for equal-length sequences, either orientation; indels/shifted alignment not screened"
        records = audit.pop("membership")
        path = OUT / f"{task}_membership.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
        audit["membership_path"] = str(path.relative_to(ROOT))
        audit["membership_sha256"] = digest(path)
        audit["chromosomes"] = {split: sorted({r["chromosome"] for r in records if r["split"] == split and r["chromosome"]}) for split in parts}
        report["splits"][task] = audit
        cache_path = ROOT / "results/activations" / f"{task}_test_residual_mean.npz"
        with np.load(cache_path, allow_pickle=True) as payload:
            labels = payload["labels"].astype(int)
            sequences = payload["sequences"]
            indices = _matched_gc_indices(sequences, labels, seed=1729, caliper=None)
            report["gc_matching"][task] = dict(full_count=len(labels), matched_count=len(indices),
                full_class_balance=dict(Counter(map(int, labels))),
                matched_class_balance=dict(Counter(map(int, labels[indices]))),
                is_full_set=bool(np.array_equal(np.sort(indices), np.arange(len(labels)))),
                selected_indices=indices.tolist(),
                full_membership_hash=hashlib.sha256("\n".join(sorted(sequence_hash(s) for s in sequences)).encode()).hexdigest(),
                matched_membership_hash=hashlib.sha256("\n".join(sorted(sequence_hash(sequences[i]) for i in indices)).encode()).hexdigest())
        report["inputs"][str(cache_path.relative_to(ROOT))] = digest(cache_path)
        pair_path = ROOT / "results/counterfactuals" / f"{task}_batch_activation_patching_pairs.tsv"
        if pair_path.exists():
            pairs = pd.read_csv(pair_path, sep="\t")
            split_hashes = {split: {sequence_hash(r["sequence"]) for r in rows} for split, rows in parts.items()}
            classified = Counter()
            for pair in pairs.itertuples():
                found = [split for split, hashes in split_hashes.items() if sequence_hash(pair.clean_sequence) in hashes]
                classified["+".join(found) or "unknown"] += 1
            report["patching"][task] = dict(pairs=len(pairs), unique_clean_sequences=pairs.clean_sequence.nunique(),
                split_membership=dict(classified), duplicate_sequence_ids=int(pairs.sequence_id.duplicated().sum()),
                scalar_target="trained standardized logistic residual probe decision function; not native model output",
                pair_distributions_status="not saved in submitted run; means cannot reconstruct pair-level uncertainty")
            report["inputs"][str(pair_path.relative_to(ROOT))] = digest(pair_path)
    token_path = ROOT / "results/enrichment/ctcf_qk_alignment_token_motif_scores.csv"
    token_table = pd.read_csv(token_path)
    report["motif_count_stages"] = count_stages(token_table)
    report["inputs"][str(token_path.relative_to(ROOT))] = digest(token_path)
    report["ctcf_methods"] = dict(query_positions="all non-padding positions including special tokens",
        qk_aggregation="mean bilinear QK logit over query positions; Pearson pooled across finite key positions",
        attention="softmax reconstructed from weight-only QK; biases/positional terms not validated against native attention",
        background="nearest token positions outside each hit, matching token count only; may include other hits or special tokens",
        aggregation_unit="historical tokens/hit occurrences, not independent sequences",
        pwm_threshold="PSSM minimum + 0.8 * (maximum-minimum), pseudocount 0.5; both strands; heuristic, no preregistration evidence",
        inference="historical token-level Pearson p-values are unsuitable for sequence-level inference; withdraw significance claims")
    metric_rows = []
    for path in sorted((ROOT / "results").rglob("*.csv")):
        if OUT in path.parents or "token_motif_scores" in path.name:
            continue
        table = pd.read_csv(path)
        for column in table.columns:
            metric = column if column in ["pearson_r", "spearman_rho", "p_value", "auroc", "auprc", "accuracy"] else None
            if column == "rho":
                metric = "attention_enrichment_ratio"
            if metric:
                validate_metric(metric, pd.to_numeric(table[column], errors="coerce"))
                metric_rows.append(dict(path=str(path.relative_to(ROOT)), column=column, definition=metric, status="range_valid"))
        report["inputs"][str(path.relative_to(ROOT))] = digest(path)
    report["metric_audit"] = metric_rows
    report["nt_comparison"] = dict(checkpoint="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        revision="not pinned in submitted configuration", pooling="mean over attention mask including special tokens",
        truncation="enabled, model default when max length is None", layer="zero-based encoder layer 11",
        rho_definition="attention enrichment ratio a_motif/a_bg, not Spearman correlation",
        status="removed from revised manuscript pending native attention, tokenizer alignment, and checkpoint validation")
    report["seconds"] = time.perf_counter() - started
    (OUT / "correctness_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"seconds":report["seconds"], "counts":report["motif_count_stages"],
                      "patching":report["patching"], "gc_matching":{k:{x:v[x] for x in ["full_count","matched_count","is_full_set"]} for k,v in report["gc_matching"].items()},
                      "split_issues":{k:{x:len(v[x]) for x in ["shared_chromosomes","cross_split_sequence_or_reverse_complement_duplicates","cross_split_overlapping_windows"]} for k,v in report["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
