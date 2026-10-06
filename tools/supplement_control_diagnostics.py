"""Diagnose inspected GSE81945 windows and verify the rare strict expansion case."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
from pyfaidx import Fasta
from transformers import AutoTokenizer
from src.adastra_candidates import allele_window,sequence_control
from src.config import DEFAULT_CONFIG
from src.control_diagnostics import diagnose_window
from src.controlled_edits import MotifDefinition
from src.motif_scoring import load_jaspar_ctcf_motif,motif_pssm
from src.utils import write_json,sha256_file
from src.variant_assay import validate_variant
from src.variant_protocol import VariantProtocol

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/"results/control_diagnostics/supplement"
    output.mkdir(parents=True,exist_ok=False)
    source=ROOT/"results/native_followup/ctcf_allele_effects.csv"
    write_json(output/"protocol.json",dict(role="post-native exploratory sequence-only supplement; no model scores",
        source_sha256=sha256_file(source),population="all 16 inspected hg19 published rows; label singleton heterozygous subset",
        analysis="same independent predicates as v1 diagnostic; no native protocol or membership changes",
        extra="verify predicted strict original-rule retention in expanded ADASTRA against original sequence_control implementation"))
    token=AutoTokenizer.from_pretrained(DEFAULT_CONFIG.model.model_name,revision=DEFAULT_CONFIG.model.revision,
        trust_remote_code=True,local_files_only=True)
    motif=MotifDefinition("CTCF",pssm=motif_pssm(load_jaspar_ctcf_motif()),fraction=.8)
    table=pd.read_csv(source);sizes=table.groupby("locus_group").size()
    rows=[];candidates=[]
    for r in table.to_dict("records"):
        index=validate_variant(r)
        values,predicates=diagnose_window(token,{**r,"variant_index":index},motif,VariantProtocol())
        rows.append(dict(source_row=r["source_row"],reference_verified=True,
            singleton_heterozygous=r["zygosity"]=="heterozygous" and sizes[r["locus_group"]]==1,**values))
        candidates.extend(dict(source_row=r["source_row"],**p) for p in predicates)
    pd.DataFrame(rows).to_csv(output/"gse_cases.csv",index=False,lineterminator="\n")
    pd.DataFrame(candidates).to_csv(output/"gse_candidates.csv",index=False,lineterminator="\n")
    diag=pd.read_csv(ROOT/"results/control_diagnostics/cases.csv")
    members=pd.read_csv(ROOT/"results/control_diagnostics/membership.csv")
    strict=[]
    with Fasta(str(ROOT/"data/genomes/hg38.fa"),as_raw=True,sequence_always_upper=True,rebuild=False) as genome:
        for v in diag.loc[diag.original,"variant_id"]:
            r=members[members.variant_id==v].iloc[0].to_dict()
            window,_=allele_window(genome,r)
            case,reason,trace=sequence_control(token,window,motif,VariantProtocol())
            assert case is not None
            strict.append(dict(variant_id=v,reason=reason,sham_index=case["sham_index"],trace=trace))
    write_json(output/"summary.json",dict(gse_source_rows=16,singleton_heterozygous=int(sum(r["singleton_heterozygous"] for r in rows)),
        gse_singleton_predicate_counts={k:sum(r[k] for r in rows if r["singleton_heterozygous"]) for k in
            ("full_allele_offsets","reference_motif","query_correspondence","original","local_only","local_no_width")},
        strict_expansion_verification=strict))
    write_json(output/"execution.json",dict(status="completed",source_code_sha256=sha256_file(Path(__file__)),
        artifacts={p.name:sha256_file(p) for p in output.iterdir() if p.name!="execution.json"}))
    print(json.dumps(json.loads((output/"summary.json").read_text()),indent=2))


if __name__=="__main__":main()
