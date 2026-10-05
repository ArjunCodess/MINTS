"""Pinned accessibility-conditioned CTCF case/control construction."""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .integrity import sequence_hash
from .probing import _sequence_gc_fraction
from .utils import write_json, sha256_file


ACCESSIBILITY_FILE = "ENCFF598KWZ"
ACCESSIBILITY_EXPERIMENT = "ENCSR000EMT"
ACCESSIBILITY_MD5 = "02802ecaf3d6a6d7891f2816bdd42421"
ACCESSIBILITY_URL = f"https://www.encodeproject.org/files/{ACCESSIBILITY_FILE}/@@download/{ACCESSIBILITY_FILE}.bed.gz"


def download_accessibility(directory):
    """Retrieve a small GRCh38 DNase narrowPeak file, verified against ENCODE."""
    import requests
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{ACCESSIBILITY_FILE}.bed.gz"
    if not path.exists():
        response = requests.get(ACCESSIBILITY_URL, timeout=90)
        response.raise_for_status()
        temporary = path.with_suffix(".download")
        temporary.write_bytes(response.content)
        if hashlib.md5(response.content).hexdigest() != ACCESSIBILITY_MD5:
            temporary.unlink()
            raise ValueError("ENCODE accessibility download checksum differs from the pinned source")
        temporary.replace(path)
    if hashlib.md5(path.read_bytes()).hexdigest() != ACCESSIBILITY_MD5:
        raise ValueError("Pinned accessibility input has changed")
    write_json(directory / "accessibility_source.json", dict(accession=ACCESSIBILITY_FILE,
               experiment=ACCESSIBILITY_EXPERIMENT, url=ACCESSIBILITY_URL, md5=ACCESSIBILITY_MD5,
               sha256=sha256_file(path), assembly="GRCh38", assay="DNase-seq", biosample="GM12878",
               label_boundary="absence of overlap with this CTCF peak file is not verified absence of binding"))
    return path


def read_accessibility(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        table = pd.read_csv(handle, sep="\t", header=None)
    if table.shape[1] < 10:
        raise ValueError("Expected ENCODE narrowPeak with signal and summit fields")
    table = table.iloc[:, :10]
    table.columns = ["chrom", "start", "end", "source_name", "score", "strand", "signal", "p", "q", "summit"]
    table = table[table.chrom.str.match(r"^chr(?:[1-9]|1[0-9]|2[0-2]|X|Y)$")].copy()
    table["center"] = np.where(table.summit >= 0, table.start + table.summit, (table.start + table.end) // 2)
    return table


def overlaps_any(chrom, start, end, indexed):
    """Interval index stores sorted starts and prefix maximum ends."""
    if chrom not in indexed:
        return False
    starts, max_ends = indexed[chrom]
    stop = np.searchsorted(starts, end, side="left")
    return bool(stop and max_ends[stop - 1] > start)


def interval_index(table):
    result = {}
    for chrom, rows in table.groupby("chrom"):
        rows = rows.sort_values("start")
        result[str(chrom)] = (rows.start.to_numpy(), np.maximum.accumulate(rows.end.to_numpy()))
    return result


def build_ctcf_case_control(ctcf_path, accessibility_path, fasta_path, output, motif, width=204,
                            per_class_train=512, per_class_validation=128, per_class_test=128, seed=1729):
    """Match accessible CTCF peaks to accessible non-overlap windows by chromosome,
    exact width, GC within .02, and log DNase signal within .5. Each pair stays
    on the same chromosome; train/validation/test use disjoint chromosomes.
    """
    from pyfaidx import Fasta
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ctcf = pd.read_csv(ctcf_path, sep="\t")
    accessibility = read_accessibility(accessibility_path)
    binding_index = interval_index(ctcf)
    fasta = Fasta(str(fasta_path), rebuild=False)
    rng = np.random.default_rng(seed)
    seen = set()
    candidates = []
    try:
        for chrom, rows in accessibility.groupby("chrom", sort=True):
            if chrom not in fasta:
                continue
            chromosome_length = len(fasta[chrom])
            for row in rows.itertuples():
                start = int(row.center) - width // 2
                end = start + width
                if start < 0 or end > chromosome_length:
                    continue
                sequence = str(fasta[chrom][start:end]).upper()
                if set(sequence) - set("ACGT"):
                    continue
                identity = sequence_hash(sequence)
                if identity in seen:
                    continue
                seen.add(identity)
                binding = overlaps_any(chrom, start, end, binding_index)
                hits = motif.hits(sequence)
                partition = "test" if chrom in {"chr20", "chr21"} else "validation" if chrom in {"chr18", "chr19"} else "train"
                candidates.append(dict(name=f"{chrom}:{start}-{end}", chrom=chrom, start=start, end=end,
                    sequence=sequence, label=int(binding), partition=partition,
                    gc=_sequence_gc_fraction(sequence), accessibility_signal=float(row.signal),
                    accessibility_source=str(row.source_name), motif_present=bool(hits),
                    motif_score=max((s for _, _, s in hits), default=float("nan"))))
    finally:
        fasta.close()
    table = pd.DataFrame(candidates)
    if table.empty:
        raise ValueError("No canonical accessibility windows were extracted")
    selected = []
    caps = dict(train=per_class_train, validation=per_class_validation, test=per_class_test)
    for partition, cap in caps.items():
        source = table[table.partition == partition]
        positive = source[source.label == 1].iloc[rng.permutation((source.label == 1).sum())]
        available = set(source.index[source.label == 0])
        count = 0
        for i, row in positive.iterrows():
            negative = table.loc[sorted(available)]
            log_difference = np.abs(np.log1p(negative.accessibility_signal.clip(lower=0)) - np.log1p(max(0, row.accessibility_signal)))
            eligible = negative[(negative.chrom == row.chrom) & (np.abs(negative.gc - row.gc) <= .02) & (log_difference <= .5)]
            if eligible.empty:
                continue
            cost = np.abs(eligible.gc - row.gc) + np.abs(np.log1p(eligible.accessibility_signal.clip(lower=0)) - np.log1p(max(0,row.accessibility_signal)))
            j = int(cost.idxmin())
            available.remove(j)
            pair_id = f"{partition}:{count}"
            for index in (i, j):
                selected.append({**table.loc[index].to_dict(), "pair_id": pair_id})
            count += 1
            if count >= cap:
                break
    result = pd.DataFrame(selected)
    if any(result[result.partition == p].label.nunique() < 2 for p in caps):
        raise ValueError("Accessibility matching failed to retain both labels in every chromosome partition")
    result.to_csv(output / "ctcf_case_control.tsv", sep="\t", index=False)
    diagnostics = []
    for (partition, label, present), group in result.groupby(["partition", "label", "motif_present"]):
        diagnostics.append(dict(partition=partition,label=int(label),motif_present=bool(present),count=len(group),
                                mean_gc=float(group.gc.mean()),mean_dnase_signal=float(group.accessibility_signal.mean())))
    pd.DataFrame(diagnostics).to_csv(output / "ctcf_cohort_balance.csv", index=False)
    write_json(output / "ctcf_cohort_manifest.json", dict(seed=seed,width=width,caps=caps,
               counts=result.groupby(["partition","label"]).size().rename("count").reset_index().to_dict("records"),
               eligible_counts=table.groupby(["partition","label"]).size().rename("count").reset_index().to_dict("records"),
               matching="same chromosome; width 204bp; GC caliper .02; log1p DNase signal caliper .5; without replacement",
               label="overlap with saved GM12878 CTCF peaks within independent DNase accessible windows",
               partition="train excludes 18/19/20/21; validation 18/19; test 20/21",
               annotations_not_available=["repeat annotation", "quantitative CTCF occupancy", "cell-type replication"],
               sources={str(p):sha256_file(Path(p)) for p in [ctcf_path,accessibility_path]},
               fasta_sha256=sha256_file(Path(fasta_path)),
               claim_boundary="accessibility-conditioned observed peak overlap, not experimentally verified binding absence"))
    return result
