"""Recover and validate Table S3 counts and hg19 reference alleles."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import subprocess

import pandas as pd
import requests

from src.native_alleles import extract_alleles
from src.utils import write_json,sha256_file

ROOT=Path(__file__).resolve().parents[1]


def run(output):
    output=Path(output)
    sources=output/"sources"
    if (output/"allele_manifest.json").exists():
        raise FileExistsError("Allele extraction outputs are immutable; use a fresh directory")
    sources.mkdir(parents=True,exist_ok=True)
    base="https://ars.els-cdn.com/content/image/1-s2.0-S221112471631628X-"
    retrieval=[]
    for name in ("mmc1.pdf","mmc2.xlsx"):
        path=sources/name
        if not path.exists():
            response=requests.get(base+name,timeout=30)
            response.raise_for_status()
            path.write_bytes(response.content)
        retrieval.append(dict(url=base+name,path=path.relative_to(ROOT).as_posix(),sha256=sha256_file(path)))
    text_path=sources/"supplement_layout.txt"
    subprocess.run(["pdftotext","-layout",str(sources/"mmc1.pdf"),str(text_path)],check=True)
    rows=extract_alleles(text_path.read_text(encoding="utf-8"))
    if len(rows)!=16:
        raise ValueError("Table S3 should contain 16 mutation rows; inspect extraction")
    def fetch(row):
        position=row["position_1based"]
        start,end=position-102,position+102
        url=f"https://api.genome.ucsc.edu/getData/sequence?genome=hg19;chrom={row['chrom']};start={start};end={end}"
        response=requests.get(url,timeout=45)
        response.raise_for_status()
        result=response.json()
        sequence=result["dna"].upper()
        if len(sequence)!=204 or sequence[101]!=row["reference"]:
            raise ValueError(f"hg19 reference/coordinate mismatch at {row['chrom']}:{position}")
        return dict(**row,window_start_0based=start,window_end_0based=end,
                    reference_sequence=sequence,alternate_sequence=sequence[:101]+row["alternate"]+sequence[102:],
                    reference_verified=True,reference_source_url=url,
                    reference_sequence_sha256=__import__("hashlib").sha256(sequence.encode()).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as executor:
        prepared=list(executor.map(fetch,rows))
    table=pd.DataFrame(prepared)
    table.to_csv(output/"ctcf_allele_effects.csv",index=False)
    identifiable=table[table.status!="not_identifiable"]
    strict=identifiable[identifiable.strict_motif_position.notna()]
    counts=table.groupby("locus_group").size()
    write_json(output/"allele_manifest.json",dict(status="prepared exploratory biological benchmark",source_table="Table S3 in mmc1.pdf",
        source_table_correction="main text references Table S2; supplementary Table S2 is RAD21 accessions",
        sources=retrieval,table_sha256=sha256_file(output/"ctcf_allele_effects.csv"),
        parser_sha256=sha256_file(ROOT/"src/native_alleles.py"),runner_sha256=sha256_file(Path(__file__)),
        variants=len(table),heterozygous_variants=len(identifiable),heterozygous_loci=int(identifiable.locus_group.nunique()),
        strict_core_variants=len(strict),strict_core_loci=int(strict.locus_group.nunique()),
        adjacent_multi_variant_loci=int((counts>1).sum()),genome_build="hg19",reference_verified=int(table.reference_verified.sum()),
        coordinates="source positions treated as 1-based and checked against UCSC hg19; windows 0-based half-open",
        count_normalization="MT_ChIP * WT_WGS / MT_WGS divided by adjusted_MT_ChIP + WT_ChIP",
        interval="half-count corrected log odds ratio +/- 1.96*sqrt(sum reciprocal counts); descriptive pooled reads",
        limitations=["replicates pooled in publication; replicate-specific variation unavailable",
                     "nearby variants may be one haplotype; phase unresolved",
                     "homozygous locus excluded from normalized allele inference",
                     "small selected cohort; no broad biological confirmation or native model claim",
                     "no substitution-matched control population yet; no hg38 liftover claimed",
                     "uncertainty excludes mapping bias, cell-line context and WGS/ChIP replicate variation"],
        raw_source_license="publisher supplemental material; third-party terms, not MINTS MIT license"))
    print(json.dumps(dict(variants=len(table),heterozygous=len(identifiable),loci=int(identifiable.locus_group.nunique())),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/native_followup")
    run(parser.parse_args().output)
