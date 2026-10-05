"""Extract published allele counts without inferring unavailable replicate data."""
from __future__ import annotations

import math
import re


ROW = re.compile(r"(chr\w+):([\d,]+)\s+([ACGT])>([ACGT])\s+(-|\d+)\s+(\d+)\s+"
                 r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)%\s+(heterozygous|homozygous)")


def allele_effect(wgs_wt, wgs_mt, chip_wt, chip_mt):
    if min(wgs_wt,wgs_mt,chip_wt,chip_mt)<0:
        raise ValueError("Allele counts must be nonnegative")
    if min(wgs_wt,wgs_mt)==0 or chip_wt+chip_mt==0:
        return dict(status="not_identifiable",normalized_vaf=None,log_odds_ratio=None)
    adjusted_mt=chip_mt*wgs_wt/wgs_mt
    normalized=adjusted_mt/(adjusted_mt+chip_wt)
    # Half-count correction gives a finite descriptive interval at zero reads.
    counts=[chip_mt+.5,chip_wt+.5,wgs_mt+.5,wgs_wt+.5]
    effect=math.log(counts[0]/counts[1])-math.log(counts[2]/counts[3])
    se=math.sqrt(sum(1/c for c in counts))
    return dict(status="descriptive_pooled_counts",normalized_vaf=normalized,
                log_odds_ratio=effect,ci_low=effect-1.96*se,ci_high=effect+1.96*se,
                interval_scope="approximate count sampling only; no replicate or mapping-bias uncertainty")


def extract_alleles(text):
    start=text.index("Table S3: COLO829") if "Table S3: COLO829" in text else text.index("Table S3:    COLO829")
    end=text.index("WGS = whole-genome",start)
    matches=list(ROW.finditer(text[start:end]))
    if not matches:
        raise ValueError("No allele table rows found")
    rows=[]
    for index,match in enumerate(matches):
        chrom,position,ref,alt,strict,consensus,ww,wm,cw,cm,vaf,zygosity=match.groups()
        counts=list(map(int,(ww,wm,cw,cm)))
        effect=allele_effect(*counts)
        if effect["normalized_vaf"] is not None and abs(effect["normalized_vaf"]-int(vaf)/100)>.011:
            raise ValueError("Published rounded normVAF differs from reconstructed counts")
        rows.append(dict(source_row=index+1,genome_build="hg19",chrom=chrom,position_1based=int(position.replace(",","")),
            reference=ref,alternate=alt,strict_motif_position=None if strict=="-" else int(strict),
            consensus_motif_position=int(consensus),wgs_wt=counts[0],wgs_mt=counts[1],
            chip_wt=counts[2],chip_mt=counts[3],reported_normalized_vaf=int(vaf)/100,
            zygosity=zygosity,**effect))
    # Nearby rows can describe one multi-base allele; phase is not supplied.
    groups={}
    previous={}
    for row in sorted(rows,key=lambda r:(r["chrom"],r["position_1based"])):
        chrom,pos=row["chrom"],row["position_1based"]
        if chrom not in previous or pos-previous[chrom]>19:
            groups[chrom]=f"{chrom}:{pos}"
        row["locus_group"]=groups[chrom]
        previous[chrom]=pos
    return rows
