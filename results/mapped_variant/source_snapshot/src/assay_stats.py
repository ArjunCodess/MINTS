"""Paired, cluster-aware estimates with explicit exploratory selection scope."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .inference import holm_adjust
from .integrity import COORD, sequence_hash


def genomic_clusters(names, sequences=None, block_bp=1_000_000):
    """Join pairs sharing genomic blocks, overlapping windows, or exact/RC inputs.

    A pair with two loci must share a resampling unit with either locus's block.
    This conservative transitive union avoids splitting matched pairs. Homology
    beyond exact/RC identity remains a separate alignment audit.
    """
    if block_bp <= 0:
        raise ValueError("block_bp must be positive")
    names = np.asarray(names, dtype=str)
    if names.ndim == 1:
        names = names[:, None]
    if names.ndim != 2 or not len(names):
        raise ValueError("Expected nonempty sequence or pair names")
    parent = list(range(len(names)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(i, j):
        parent[find(j)] = find(i)
    owners = {}
    intervals = {}
    for i, row in enumerate(names):
        for name in row:
            match = COORD.search(name)
            if not match:
                raise ValueError(f"Missing genomic coordinates: {name}")
            chrom, start, end = match[1].lower(), int(match[2]), int(match[3])
            if end <= start:
                raise ValueError(f"Invalid genomic interval: {name}")
            intervals.setdefault(chrom, []).append((start, end, i))
            for block in range(start // block_bp, (end - 1) // block_bp + 1):
                key = (chrom, block)
                if key in owners:
                    join(i, owners[key])
                owners[key] = i
    for rows in intervals.values():
        active = []
        for start, end, i in sorted(rows):
            active = [(b, j) for b, j in active if b > start]
            for _, j in active:
                join(i, j)
            active.append((end, i))
    if sequences is not None:
        sequences = np.asarray(sequences, dtype=str)
        if sequences.ndim == 1:
            sequences = sequences[:, None]
        if sequences.shape != names.shape:
            raise ValueError("Sequence identities must align with genomic names")
        identities = {}
        for i, row in enumerate(sequences):
            for sequence in row:
                key = sequence_hash(sequence)
                if key in identities:
                    join(i, identities[key])
                identities[key] = i
    codes = {}
    return np.asarray([codes.setdefault(find(i), len(codes)) for i in range(len(names))])


def cluster_draws(clusters, rng, repetitions):
    clusters = np.asarray(clusters)
    groups = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    if len(groups) < 2:
        raise ValueError("At least two independent resampling clusters are required")
    for _ in range(repetitions):
        yield np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])


def paired_auc_difference(labels, first, second, clusters=None, repetitions=2000, seed=1729):
    y, a, b = map(np.asarray, (labels, first, second))
    if y.ndim != 1 or y.shape != a.shape or y.shape != b.shape or np.unique(y).size != 2:
        raise ValueError("Paired predictions must align and contain both classes")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or repetitions < 1:
        raise ValueError("Expected finite predictions and positive repetitions")
    clusters = np.arange(len(y)) if clusters is None else np.asarray(clusters)
    if clusters.shape != y.shape:
        raise ValueError("Clusters must align with predictions")
    observed = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    draws = []
    for index in cluster_draws(clusters, np.random.default_rng(seed), repetitions):
        if np.unique(y[index]).size == 2:
            draws.append(roc_auc_score(y[index], a[index]) - roc_auc_score(y[index], b[index]))
    if len(draws) < max(2, repetitions // 2):
        raise ValueError("Insufficient valid two-class bootstrap draws")
    low, high = np.quantile(draws, [.025, .975])
    return dict(difference=observed, ci_low=float(low), ci_high=float(high),
                valid_draws=len(draws), clusters=int(np.unique(clusters).size),
                estimand="paired held-out AUROC difference conditional on fixed classifiers")


def paired_head_inference(differences, clusters=None, repetitions=19999, bootstrap_samples=2000, seed=1729):
    """Cluster sign flips and simultaneous max-statistic inference over all heads.

    Cluster sums are flipped together. Exchangeability is still an assumption
    for observational comparisons. Max tests use studentized statistics so
    unequal head variances do not favor large-scale heads.
    """
    x = np.asarray(differences, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not np.isfinite(x).all() or len(x) < 2:
        raise ValueError("Expected finite pair-by-head differences")
    clusters = np.arange(len(x)) if clusters is None else np.asarray(clusters)
    if clusters.shape != (len(x),) or repetitions < 1 or bootstrap_samples < 2:
        raise ValueError("Invalid clusters or resampling counts")
    groups = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    if len(groups) < 2:
        raise ValueError("At least two genomic clusters are required")
    sums = np.stack([x[g].sum(axis=0) for g in groups])
    observed = x.mean(axis=0)
    scale = np.sqrt(np.sum((sums - np.asarray([len(g) for g in groups])[:, None] * observed) ** 2, axis=0)) / len(x)
    # The uncentered RMS prevents undefined null tests for constant nonzero effects.
    null_scale = np.sqrt((sums * sums).sum(axis=0)) / len(x)
    t = np.divide(np.abs(observed), null_scale, out=np.zeros_like(observed), where=null_scale > 0)
    rng = np.random.default_rng(seed)
    exceed = np.zeros(x.shape[1], dtype=int)
    max_exceed = np.zeros_like(exceed)
    for start in range(0, repetitions, 128):
        signs = rng.choice([-1., 1.], size=(min(128, repetitions - start), len(groups)))
        null = signs @ sums / len(x)
        null_t = np.divide(np.abs(null), null_scale, out=np.zeros_like(null), where=null_scale > 0)
        exceed += (null_t >= t - 1e-12).sum(axis=0)
        max_exceed += (null_t.max(axis=1)[:, None] >= t - 1e-12).sum(axis=0)
    p = (exceed + 1) / (repetitions + 1)
    draws = np.stack([x[i].mean(axis=0) for i in cluster_draws(clusters, rng, bootstrap_samples)])
    ci = np.quantile(draws, [.025, .975], axis=0)
    standardized = np.divide(np.abs(draws - observed), scale, out=np.zeros_like(draws), where=scale > 0)
    radius = float(np.quantile(standardized.max(axis=1), .95))
    return dict(mean=observed, ci_low=ci[0], ci_high=ci[1], p=p,
                holm_p=holm_adjust(p), max_stat_p=(max_exceed + 1) / (repetitions + 1),
                simultaneous_low=observed - radius * scale,
                simultaneous_high=observed + radius * scale, clusters=len(groups))


def absolute_patching_summary(clean, corrupted, patched, sequences=None, repetitions=1000, seed=1729):
    clean, corrupted, patched = map(lambda a: np.asarray(a, dtype=float), (clean, corrupted, patched))
    if clean.shape != corrupted.shape or patched.shape[0] != len(clean):
        raise ValueError("Patching scores must align")
    effects = patched - corrupted.reshape((-1,) + (1,) * (patched.ndim - 1))
    flat = effects.reshape(len(clean), -1)
    clusters = np.arange(len(clean)) if sequences is None else np.unique(np.asarray(sequences), return_inverse=True)[1]
    if not len(clean):
        raise ValueError("At least one pair is required")
    means = flat.mean(axis=0)
    loo = (flat.sum(axis=0)[None, :] - flat) / (len(flat) - 1) if len(flat) > 1 else np.full_like(flat, np.nan)
    if np.unique(clusters).size >= 2:
        draws = np.stack([flat[i].mean(axis=0) for i in cluster_draws(clusters, np.random.default_rng(seed), repetitions)])
        ci = np.quantile(draws, [.025, .975], axis=0)
    else:
        ci = np.full((2, flat.shape[1]), np.nan)
    return dict(mean=means, median=np.median(flat, axis=0), ci_low=ci[0], ci_high=ci[1],
                loo_low=loo.min(axis=0), loo_high=loo.max(axis=0),
                positive_corruption_pairs=int((clean > corrupted).sum()),
                negative_corruption_pairs=int((clean < corrupted).sum()), effects=effects)
