"""Fail-closed split checks and portable membership records."""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict

COORD = re.compile(r"(chr[\w]+):(\d+)-(\d+)", re.I)


def sequence_hash(sequence):
    text = str(sequence).upper()
    reverse = text.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
    return hashlib.sha256(min(text, reverse).encode()).hexdigest()


def membership(splits):
    records = []
    for split, rows in splits.items():
        for index, row in enumerate(rows):
            text = str(row["sequence"]).upper()
            match = COORD.search(str(row.get("name", "")))
            records.append(dict(split=split, index=index, name=str(row.get("name", "")),
                                label=int(row["label"]), sequence_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                canonical_sequence_sha256=sequence_hash(text), length=len(text),
                                chromosome=match[1].lower() if match else None,
                                start=int(match[2]) if match else None, end=int(match[3]) if match else None))
    return records


def audit_splits(splits):
    records = membership(splits)
    hashes, chromosomes = defaultdict(set), defaultdict(set)
    intervals = defaultdict(list)
    for row in records:
        hashes[row["canonical_sequence_sha256"]].add(row["split"])
        if row["chromosome"]:
            chromosomes[row["chromosome"]].add(row["split"])
            intervals[row["chromosome"]].append(row)
    duplicates = [key for key, parts in hashes.items() if len(parts) > 1]
    shared_chromosomes = [key for key, parts in chromosomes.items() if len(parts) > 1]
    overlaps = []
    for chrom, rows in intervals.items():
        active = []
        for row in sorted(rows, key=lambda x: x["start"]):
            active = [other for other in active if other["end"] > row["start"]]
            for other in active:
                if other["split"] != row["split"]:
                    overlaps.append([other["name"], row["name"]])
            active.append(row)
    counts = {split: dict(Counter(int(r["label"]) for r in rows)) for split, rows in splits.items()}
    return dict(class_counts=counts, shared_chromosomes=shared_chromosomes,
                cross_split_sequence_or_reverse_complement_duplicates=duplicates,
                cross_split_overlapping_windows=overlaps,
                missing_coordinates=sum(r["chromosome"] is None for r in records),
                partitions=list(splits), membership=records,
                near_duplicate_status="not yet audited; exact/RC and interval checks do not certify sequence similarity")


def assert_disjoint_splits(splits):
    report = audit_splits(splits)
    if report["cross_split_sequence_or_reverse_complement_duplicates"] or report["cross_split_overlapping_windows"]:
        raise ValueError("Cross-partition sequence/RC duplication or overlapping genomic windows detected")
    if report["missing_coordinates"] or report["shared_chromosomes"]:
        raise ValueError("Chromosome-disjoint partitioning cannot be certified")
    near_duplicates=near_duplicate_pairs(splits)
    if near_duplicates:
        raise ValueError("Cross-partition near-duplicate sequences detected (<=2 substitutions, either orientation)")
    report["near_duplicates"]=near_duplicates
    report["near_duplicate_status"]="exhaustive <=2 substitutions for equal-length sequences, either orientation; indels/shifted alignment not screened"
    return report


def near_duplicate_pairs(splits, max_substitutions=2):
    """Exhaustive equal-length Hamming/RC check using pigeonhole indexing.

    Split each sequence into d+1 blocks. Sequences with <=d substitutions
    share at least one block. This does not detect indels or shifted windows.
    """
    index = defaultdict(list)
    findings = []
    for split, rows in splits.items():
        pending = []
        for row_id, row in enumerate(rows):
            sequence = str(row["sequence"]).upper()
            reverse = sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
            if len(sequence) <= max_substitutions:
                raise ValueError("Sequence too short for near-duplicate indexing")
            boundaries = [len(sequence)*i//(max_substitutions+1) for i in range(max_substitutions+2)]
            candidates = set()
            for text in [sequence, reverse]:
                for block in range(max_substitutions+1):
                    candidates.update(index[(len(text), block, text[boundaries[block]:boundaries[block+1]])])
            for other_split, other_id, other in candidates:
                distance = min(sum(a != b for a,b in zip(sequence,other)), sum(a != b for a,b in zip(reverse,other)))
                if distance <= max_substitutions:
                    findings.append(dict(split=split,index=row_id,other_split=other_split,other_index=other_id,distance=distance))
            pending.append((split,row_id,sequence))
        for item in pending:
            text = item[2]
            boundaries = [len(text)*i//(max_substitutions+1) for i in range(max_substitutions+2)]
            for block in range(max_substitutions+1):
                index[(len(text),block,text[boundaries[block]:boundaries[block+1]])].append(item)
    return findings
