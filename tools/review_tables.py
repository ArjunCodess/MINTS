"""Regenerate manuscript tables and save sequence-level predictions/intervals."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from src.probing import _gc_content_features, _matched_gc_indices, bootstrap_probe_confidence_intervals, _classification_metrics
from src.config import DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/review"


def tex_table(rows, metrics):
    lines = [r"\begin{tabular}{l" + "r" * len(metrics) + "}", r"\toprule",
             "Task & " + " & ".join(metrics.values()) + r" \\", r"\midrule"]
    for task, group in rows.groupby("task", sort=False):
        entries = []
        for method in metrics:
            row = group[group.method == method].iloc[0]
            entries.append(f"{row.auroc:.4f} [{row.auroc_ci_low:.4f}, {row.auroc_ci_high:.4f}]")
        label = {"promoter_tata":"TATA promoter", "promoter_no_tata":"Other promoter",
                 "splice_sites_donors":"Splice donor", "splice_sites_acceptors":"Splice acceptor"}[task]
        lines.append(label + " & " + " & ".join(entries) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, matching = [], []
    for task in DEFAULT_CONFIG.data.task_names:
        print(task, flush=True)
        with np.load(ROOT / f"results/activations/{task}_train_residual_mean.npz", allow_pickle=True) as p:
            train_sequences = p["sequences"].astype(str)
            y_train = p["labels"].astype(int)
            x_train = p["residual_mean"][:, list(p["layers"]).index(11)].copy()
        with np.load(ROOT / f"results/activations/{task}_test_residual_mean.npz", allow_pickle=True) as p:
            sequences = p["sequences"].astype(str)
            y = p["labels"].astype(int)
            x_test = p["residual_mean"][:, list(p["layers"]).index(11)].copy()
            names = p["names"].astype(str)
        gc = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=1729)
        gc.fit(_gc_content_features(train_sequences), y_train)
        gc_prob = gc.predict_proba(_gc_content_features(sequences))[:, 1]
        kmer = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(3,6), lowercase=False, min_df=2,
                                            max_features=50000, dtype=np.float32),
                             LogisticRegression(max_iter=1000, class_weight="balanced", random_state=1729, solver="saga"))
        kmer.fit(train_sequences.tolist(), y_train)
        kmer_prob = kmer.predict_proba(sequences.tolist())[:, 1]
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=1729))
        probe.fit(x_train, y_train)
        probe_prob = probe.predict_proba(x_test)[:, 1]
        historical = _matched_gc_indices(sequences, y, seed=1729, caliper=None)
        matched = _matched_gc_indices(sequences, y, seed=1729, caliper=0.02)
        features = _gc_content_features(sequences)[:,0]
        matching.append(dict(task=task, full_count=len(y), historical_count=len(historical), caliper_count=len(matched),
            caliper=0.02, full_positive_mean_gc=float(features[y==1].mean()), full_negative_mean_gc=float(features[y==0].mean()),
            matched_positive_mean_gc=float(features[matched][y[matched]==1].mean()) if len(matched) else None,
            matched_negative_mean_gc=float(features[matched][y[matched]==0].mean()) if len(matched) else None))
        pred = pd.DataFrame(dict(name=names, label=y, gc_probability=gc_prob, kmer_probability=kmer_prob,
                                 probe_probability=probe_prob, gc_caliper_selected=np.isin(np.arange(len(y)), matched),
                                 historical_gc_selected=np.isin(np.arange(len(y)), historical)))
        pred.to_csv(OUT / f"{task}_predictions.csv", index=False)
        for method, prob, indices in [("gc",gc_prob,np.arange(len(y))), ("kmer",kmer_prob,np.arange(len(y))),
                                      ("probe",probe_prob,np.arange(len(y))), ("historical_gc_probe",probe_prob,historical),
                                      ("caliper_gc_probe",probe_prob,matched), ("caliper_gc_baseline",gc_prob,matched)]:
            if len(indices) == 0:
                continue
            sample_y, sample_prob = y[indices], prob[indices]
            metric = _classification_metrics(sample_y, sample_prob, (sample_prob>=0.5).astype(int))
            ci = bootstrap_probe_confidence_intervals(sample_y, sample_prob, (sample_prob>=0.5).astype(int),seed=1729)
            row = dict(task=task, method=method, train_examples=len(y_train), test_examples=len(indices), **metric)
            for key, (low, high) in ci.items():
                row[f"{key}_ci_low"], row[f"{key}_ci_high"] = low, high
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "classification_metrics.csv", index=False)
    (OUT / "gc_matching_diagnostics.json").write_text(json.dumps(matching, indent=2),encoding="utf-8")
    (OUT / "table3.tex").write_text(tex_table(table, dict(gc="GC",kmer="$3$--$6$-mer",probe="Frozen readout")),encoding="utf-8")
    (OUT / "table4.tex").write_text(tex_table(table, dict(probe="Full",historical_gc_probe="Historical matching",caliper_gc_probe="GC caliper")),encoding="utf-8")
    manifest = dict(command=".venv/Scripts/python.exe tools/review_tables.py", seed=1729, bootstrap_samples=1000,
                    bootstrap_unit="test sequence; fixed trained model", confidence_level=0.95, seconds=time.perf_counter()-started,
                    caliper_status="0.02 chosen heuristically during October review, not preregistered",
                    artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*") if p.suffix in [".csv",".tex"]})
    (OUT / "tables_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
