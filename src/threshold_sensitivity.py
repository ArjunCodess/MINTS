"""Threshold sensitivity and null-calibration summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .utils import progress, utc_now_iso


QK_R_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)
RHO_THRESHOLDS = (1.1, 1.25, 1.5, 2.0)
PM_THRESHOLDS = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class SensitivityOutputs:
    """Paths written by the sensitivity export."""

    table: Path
    figure: Path
    manifest: Path


def _best_head(table: pd.DataFrame, metric: str) -> tuple[int | None, int | None, float]:
    values = pd.to_numeric(table[metric], errors="coerce")
    if values.notna().sum() == 0:
        return None, None, float("nan")
    idx = int(values.idxmax())
    row = table.loc[idx]
    return int(row["layer"]), int(row["head"]), float(row[metric])


def _load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required sensitivity input is missing: {path}")
    return pd.read_csv(path)


def _qk_rows(qk: pd.DataFrame) -> list[dict[str, object]]:
    best_layer, best_head, best_value = _best_head(qk, "pearson_r")
    rows: list[dict[str, object]] = []
    for threshold in QK_R_THRESHOLDS:
        passing = qk[pd.to_numeric(qk["pearson_r"], errors="coerce") >= threshold]
        rows.append(
            {
                "analysis": "single_metric_sensitivity",
                "target": "CTCF",
                "metric": "qk_pearson_r",
                "r_threshold": threshold,
                "rho_threshold": np.nan,
                "pm_threshold": np.nan,
                "observed_pass_count": int(len(passing)),
                "qk_pass_count": int(len(passing)),
                "enrichment_pass_count": np.nan,
                "patching_pass_count": np.nan,
                "joint_pass_count": np.nan,
                "null_type": "",
                "null_mean": np.nan,
                "null_p95": np.nan,
                "best_layer": best_layer,
                "best_head": best_head,
                "best_value": best_value,
                "interpretation": "QK threshold relaxation only.",
            }
        )
    return rows


def _enrichment_rows(enrichment: pd.DataFrame) -> list[dict[str, object]]:
    best_layer, best_head, best_value = _best_head(enrichment, "rho")
    rows: list[dict[str, object]] = []
    rho_values = pd.to_numeric(enrichment["rho"], errors="coerce")
    for threshold in RHO_THRESHOLDS:
        passing = enrichment[rho_values >= threshold]
        rows.append(
            {
                "analysis": "single_metric_sensitivity",
                "target": "CTCF",
                "metric": "attention_enrichment_rho",
                "r_threshold": np.nan,
                "rho_threshold": threshold,
                "pm_threshold": np.nan,
                "observed_pass_count": int(len(passing)),
                "qk_pass_count": np.nan,
                "enrichment_pass_count": int(len(passing)),
                "patching_pass_count": np.nan,
                "joint_pass_count": np.nan,
                "null_type": "matched_background_observed",
                "null_mean": float(np.nanmean(rho_values)),
                "null_p95": float(np.nanpercentile(rho_values, 95)),
                "best_layer": best_layer,
                "best_head": best_head,
                "best_value": best_value,
                "interpretation": "Matched motif/background enrichment threshold relaxation.",
            }
        )
    return rows


def _patching_rows(config: PipelineConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task, filename in {
        "promoter_tata": "promoter_tata_batch_dnabert_activation_patching.csv",
        "splice_sites_donors": "splice_sites_donors_batch_dnabert_activation_patching.csv",
    }.items():
        path = config.paths.patching_dir / filename
        if not path.exists():
            continue
        table = pd.read_csv(path)
        best_layer, best_head, best_value = _best_head(table, "restoration")
        restoration = pd.to_numeric(table["restoration"], errors="coerce")
        for threshold in PM_THRESHOLDS:
            passing = table[restoration >= threshold]
            rows.append(
                {
                    "analysis": "single_metric_sensitivity",
                    "target": task,
                    "metric": "patching_pm",
                    "r_threshold": np.nan,
                    "rho_threshold": np.nan,
                    "pm_threshold": threshold,
                    "observed_pass_count": int(len(passing)),
                    "qk_pass_count": np.nan,
                    "enrichment_pass_count": np.nan,
                    "patching_pass_count": int(len(passing)),
                    "joint_pass_count": np.nan,
                    "null_type": "",
                    "null_mean": np.nan,
                    "null_p95": np.nan,
                    "best_layer": best_layer,
                    "best_head": best_head,
                    "best_value": best_value,
                    "interpretation": "Task-specific patching sensitivity; not part of the strict CTCF chain.",
                }
            )
    return rows


def _joint_ctcf_rows(qk: pd.DataFrame, enrichment: pd.DataFrame, seed: int) -> list[dict[str, object]]:
    merged = qk[["layer", "head", "pearson_r"]].merge(
        enrichment[["layer", "head", "rho"]],
        on=["layer", "head"],
        how="inner",
    )
    r_values = pd.to_numeric(merged["pearson_r"], errors="coerce").to_numpy(dtype=np.float64)
    rho_values = pd.to_numeric(merged["rho"], errors="coerce").to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for r_threshold in QK_R_THRESHOLDS:
        qk_mask = np.isfinite(r_values) & (r_values >= r_threshold)
        for rho_threshold in RHO_THRESHOLDS:
            rho_mask = np.isfinite(rho_values) & (rho_values >= rho_threshold)
            observed = int(np.sum(qk_mask & rho_mask))
            null_counts = []
            for _ in range(1000):
                permuted = rng.permutation(rho_mask)
                null_counts.append(int(np.sum(qk_mask & permuted)))
            null_array = np.asarray(null_counts, dtype=np.float64)
            rows.append(
                {
                    "analysis": "joint_ctcf_sensitivity",
                    "target": "CTCF",
                    "metric": "qk_and_enrichment",
                    "r_threshold": r_threshold,
                    "rho_threshold": rho_threshold,
                    "pm_threshold": np.nan,
                    "observed_pass_count": observed,
                    "qk_pass_count": int(np.sum(qk_mask)),
                    "enrichment_pass_count": int(np.sum(rho_mask)),
                    "patching_pass_count": np.nan,
                    "joint_pass_count": observed,
                    "null_type": "permuted_head_alignment",
                    "null_mean": float(np.mean(null_array)),
                    "null_p95": float(np.percentile(null_array, 95)),
                    "best_layer": np.nan,
                    "best_head": np.nan,
                    "best_value": np.nan,
                    "interpretation": "Joint CTCF QK/enrichment count under threshold relaxation.",
                }
            )
    return rows


def _unavailable_null_rows() -> list[dict[str, object]]:
    return [
        {
            "analysis": "null_calibration_status",
            "target": "CTCF",
            "metric": "qk_pearson_r",
            "r_threshold": np.nan,
            "rho_threshold": np.nan,
            "pm_threshold": np.nan,
            "observed_pass_count": np.nan,
            "qk_pass_count": np.nan,
            "enrichment_pass_count": np.nan,
            "patching_pass_count": np.nan,
            "joint_pass_count": np.nan,
            "null_type": "shuffled_motif_scores",
            "null_mean": np.nan,
            "null_p95": np.nan,
            "best_layer": np.nan,
            "best_head": np.nan,
            "best_value": np.nan,
            "interpretation": (
                "Not recomputed from aggregate artifacts; requires per-token QK score vectors, "
                "not just saved per-head Pearson summaries."
            ),
        },
        {
            "analysis": "null_calibration_status",
            "target": "CTCF",
            "metric": "attention_enrichment_rho",
            "r_threshold": np.nan,
            "rho_threshold": np.nan,
            "pm_threshold": np.nan,
            "observed_pass_count": np.nan,
            "qk_pass_count": np.nan,
            "enrichment_pass_count": np.nan,
            "patching_pass_count": np.nan,
            "joint_pass_count": np.nan,
            "null_type": "gc_matched_background",
            "null_mean": np.nan,
            "null_p95": np.nan,
            "best_layer": np.nan,
            "best_head": np.nan,
            "best_value": np.nan,
            "interpretation": (
                "Not present in current artifacts; current enrichment uses position-matched "
                "non-motif backgrounds."
            ),
        },
    ]


def _write_figure(table: pd.DataFrame, figure_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))

    qk = table[(table["analysis"] == "single_metric_sensitivity") & (table["metric"] == "qk_pearson_r")]
    axes[0, 0].plot(qk["r_threshold"], qk["observed_pass_count"], marker="o", color="#2f5d8c")
    axes[0, 0].set_title("CTCF QK threshold")
    axes[0, 0].set_xlabel("minimum Pearson r")
    axes[0, 0].set_ylabel("passing heads")

    enrich = table[
        (table["analysis"] == "single_metric_sensitivity")
        & (table["metric"] == "attention_enrichment_rho")
    ]
    axes[0, 1].plot(enrich["rho_threshold"], enrich["observed_pass_count"], marker="o", color="#2f7d59")
    axes[0, 1].axvline(2.0, color="#555555", linestyle="--", linewidth=1)
    axes[0, 1].set_title("CTCF enrichment threshold")
    axes[0, 1].set_xlabel("minimum rho")
    axes[0, 1].set_ylabel("passing heads")

    patch = table[(table["analysis"] == "single_metric_sensitivity") & (table["metric"] == "patching_pm")]
    for target, group in patch.groupby("target"):
        axes[1, 0].plot(group["pm_threshold"], group["observed_pass_count"], marker="o", label=target)
    axes[1, 0].set_title("Auxiliary patching thresholds")
    axes[1, 0].set_xlabel("minimum PM")
    axes[1, 0].set_ylabel("passing heads")
    axes[1, 0].legend(frameon=False, fontsize=8)

    joint = table[
        (table["analysis"] == "joint_ctcf_sensitivity")
        & (table["rho_threshold"] == 1.1)
    ].sort_values("r_threshold")
    axes[1, 1].plot(joint["r_threshold"], joint["joint_pass_count"], marker="o", color="#8f4f6f")
    axes[1, 1].fill_between(
        joint["r_threshold"].astype(float),
        0,
        joint["null_p95"].astype(float),
        color="#8f4f6f",
        alpha=0.18,
        label="permuted-head p95",
    )
    axes[1, 1].set_title("Joint CTCF count at rho >= 1.1")
    axes[1, 1].set_xlabel("minimum Pearson r")
    axes[1, 1].set_ylabel("joint passing heads")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def run_threshold_sensitivity(config: PipelineConfig = DEFAULT_CONFIG) -> SensitivityOutputs:
    """Create threshold sensitivity and null-calibration artifacts."""

    config.ensure_paths()
    progress("Loading QK, enrichment, and patching tables for threshold sensitivity")
    qk = _load_required_csv(config.paths.qk_alignment_dir / "ctcf_qk_alignment.csv")
    enrichment = _load_required_csv(config.paths.enrichment_dir / "ctcf_qk_alignment_matched_attention_enrichment.csv")
    rows: list[dict[str, object]] = []
    rows.extend(_qk_rows(qk))
    rows.extend(_enrichment_rows(enrichment))
    rows.extend(_patching_rows(config))
    rows.extend(_joint_ctcf_rows(qk, enrichment, seed=config.data.seed))
    rows.extend(_unavailable_null_rows())

    table = pd.DataFrame(rows)
    table_path = config.paths.tables_dir / "threshold_sensitivity.csv"
    figure_path = config.paths.figures_dir / "threshold_sensitivity.png"
    manifest_path = config.paths.manifests_dir / "threshold_sensitivity_manifest.json"
    table.to_csv(table_path, index=False)
    _write_figure(table, figure_path)
    manifest = {
        "created_at": utc_now_iso(),
        "table_path": str(table_path),
        "figure_path": str(figure_path),
        "qk_r_thresholds": list(QK_R_THRESHOLDS),
        "rho_thresholds": list(RHO_THRESHOLDS),
        "pm_thresholds": list(PM_THRESHOLDS),
        "null_calibration": {
            "matched_background": "observed enrichment table uses deterministic position-matched backgrounds",
            "permuted_head_alignment": "1,000 permutations of enrichment-pass labels against QK-pass labels",
            "shuffled_motif_scores": "not recomputed from aggregate artifacts",
            "gc_matched_background": "not present in current artifacts",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return SensitivityOutputs(table=table_path, figure=figure_path, manifest=manifest_path)
