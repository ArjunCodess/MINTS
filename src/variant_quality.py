"""Explicit biological QC gaps and cross-cohort dependence checks."""
import hashlib


def biological_qc(row):
    """Pooled counts never silently satisfy confirmation-level biological QC."""
    missing=[key for key in ("phase_verified", "mapping_bias_corrected", "copy_number_addressed",
                            "replicate_counts_available", "donor_id_verified", "alternate_evidence_verified")
             if row.get(key) is not True]
    return dict(confirmation_eligible=not missing,missing=missing,
                count_scope="input-normalized pooled counts are descriptive until missing checks resolved")


def sequence_identity(sequence):
    if not sequence or set(sequence)-set("ACGT"):
        raise ValueError("Nonempty canonical DNA required for sequence separation")
    rc=sequence.translate(str.maketrans("ACGT","TGCA"))[::-1]
    return hashlib.sha256(min(sequence,rc).encode()).hexdigest()


def separation_audit(discovery, confirmation):
    """Report exact/RC, shared source/donor and coordinate overlap; require homology separately."""
    reasons=[]
    if not discovery or not confirmation:
        return dict(verified=False,reasons=["both cohort memberships required"],homology_verified=False)
    for d in discovery:
        for c in confirmation:
            if d.get("genome_build") not in ("hg19","hg38") or c.get("genome_build") not in ("hg19","hg38"):
                reasons.append("genome build missing or unsupported")
            elif d["genome_build"]!=c["genome_build"]:
                reasons.append("different genome builds require verified coordinate correspondence")
            if sequence_identity(d["reference_sequence"])==sequence_identity(c["reference_sequence"]):
                reasons.append("exact/reverse-complement sequence overlap")
            if d.get("genome_build")==c.get("genome_build") and d["chrom"]==c["chrom"]:
                if d["window_start_0based"]<c["window_end_0based"] and c["window_start_0based"]<d["window_end_0based"]:
                    reasons.append("genomic interval overlap")
            for key in ("donor_id", "biosample_id", "source_experiment"):
                if not d.get(key) or not c.get(key):
                    reasons.append("missing "+key)
                elif d.get(key)==c.get(key):
                    reasons.append("shared "+key)
    return dict(verified=False,reasons=sorted(set(reasons)),
                exact_and_coordinate_separation_passed=not reasons,homology_verified=False,
                remaining="homology, missing source/donor identifiers and inspection ledger require explicit review")
