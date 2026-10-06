"""Bounded sequence-only engineering screen of the complete public CTCF table."""
from pathlib import Path,PurePosixPath
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter
from dataclasses import asdict
import json
import gzip
import time
import zipfile

import pandas as pd
from pyfaidx import Fasta
from transformers import AutoTokenizer

from src.adastra_candidates import import_candidates,select_candidates,allele_window,sequence_control
from src.config import DEFAULT_CONFIG
from src.controlled_edits import MotifDefinition
from src.motif_scoring import load_jaspar_ctcf_motif,motif_pssm,find_jaspar_matrix_path
from src.utils import sha256_file,write_json
from src.variant_protocol import VariantProtocol
from tools.download_adastra import verify_archive,MD5,SIZE,URL

ROOT=Path(__file__).resolve().parents[1]


def run(archive,output,genome_path,limit=4096):
    archive,output,genome_path=(Path(p).resolve() for p in (archive,output,genome_path))
    output.mkdir(parents=True,exist_ok=True)
    sources=["src/adastra_candidates.py","tools/run_adastra_feasibility.py","tools/download_adastra.py","tools/audit_adastra_feasibility.py",
             "src/variant_protocol.py","src/assay_alignment.py","src/native_followup.py","src/controlled_edits.py",
             "src/motif_scoring.py","src/config.py","src/utils.py"]
    hashes={n:sha256_file(ROOT/n) for n in sources}
    protocol=VariantProtocol()
    frozen=dict(version="adastra-sequence-screen-v1",role="exploratory engineering; not biological confirmation",
        release="Mabel v6.1",record="14174114",selection_limit=limit,selection_seed=1731,
        selection="ascending sha256(seed:variant_id); all source rows eligible; no FDR, effect or motif-concordance filtering",
        window_width_bp=204,variant_index=102,genome_build="hg38",source_sha256=hashes,
        sequence_constraints=asdict(protocol),model_revision=DEFAULT_CONFIG.model.revision,
        outcome_policy="source outcomes never used for eligibility, scoring, head selection or correlation",
        biological_scope="reference plus one substituted nucleotide; donor genotype and haplotype are unverified",
        stop="no native inference, head search, power or confirmation in this engineering screen")
    # Claim and freeze the design before reading the source table or any outcomes.
    with (output/"protocol.json").open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(frozen,stream,indent=2,sort_keys=True);stream.write("\n")
    started=time.time();receipt=dict(status="running",source_sha256=hashes,protocol_sha256=sha256_file(output/"protocol.json"))
    write_json(output/"execution.json",receipt)
    try:
        verify_archive(archive)
        archive_hash=sha256_file(archive)
        if not (output/"download.json").exists():
            write_json(output/"download.json",dict(status="completed",url=URL,publisher_md5=MD5,bytes=SIZE,
                sha256=archive_hash,archive_path=archive.as_posix(),retrieval="verified previously downloaded cache"))
        download=json.loads((output/"download.json").read_text())
        if download["status"]!="completed" or download["sha256"]!=archive_hash:
            raise ValueError("Download receipt does not match the verified archive")
        with zipfile.ZipFile(archive) as package:
            members=[i for i in package.infolist() if PurePosixPath(i.filename).name in ("CTCF_HUMAN.tsv","CTCF.tsv")]
            if len(members)!=1:raise ValueError("Expected exactly one complete CTCF table in the pinned archive")
            member=members[0]
            if member.file_size>256*1024*1024:raise ValueError("Unexpectedly large CTCF member")
            # Read only this member; never unpack publisher paths onto the filesystem.
            with package.open(member) as stream:raw=pd.read_csv(stream,sep="\t",dtype=str,keep_default_na=False)
        all_rows=import_candidates(raw)
        with (output/"candidates.csv.gz").open("wb") as sink:
            with gzip.GzipFile(filename="",mode="wb",fileobj=sink,mtime=0) as compressed:
                compressed.write(all_rows.to_csv(index=False,lineterminator="\n").encode())
        membership=select_candidates(all_rows,limit,1731)
        membership.to_csv(output/"membership.csv",index=False,lineterminator="\n")
        write_json(output/"source.json",dict(archive_sha256=archive_hash,member=member.filename,
            member_crc32=f"{member.CRC:08x}",uncompressed_bytes=member.file_size,rows=len(raw),columns=list(raw),
            genome_sha256=sha256_file(genome_path),motif_sha256=sha256_file(find_jaspar_matrix_path()),
            denominator="all CTCF coverage-eligible records, including nonsignificant events",
            coordinate_conversion="start is zero-based; end is one-based SNV position; end=start+1",
            effect_semantics="allele-wise weighted log2 observed/expected ratios; not pooled input-normalized binding log odds",
            outcomes_used=False))
        tokenizer=AutoTokenizer.from_pretrained(DEFAULT_CONFIG.model.model_name,revision=DEFAULT_CONFIG.model.revision,
            trust_remote_code=True,local_files_only=True)
        motif=MotifDefinition("CTCF",pssm=motif_pssm(load_jaspar_ctcf_motif()),fraction=.8)
        rows=[];traces=[];cases=[]
        with Fasta(str(genome_path),as_raw=True,sequence_always_upper=True,rebuild=False) as genome:
            for number,row in enumerate(membership.to_dict("records"),1):
                window,reason=allele_window(genome,row)
                case=None;trace=dict(candidate_positions_checked=0,candidate_rejections={})
                if window is not None:case,reason,trace=sequence_control(tokenizer,window,motif,protocol)
                rows.append(dict(**row,reference_verified=window is not None,sequence_control_feasible=case is not None,reason=reason))
                traces.append(dict(variant_id=row["variant_id"],reason=reason,**trace))
                if case is not None:cases.append(dict(**row,**window,**case,biological_confirmation_eligible=False))
                if number%256==0:print(f"screened {number}/{len(membership)}; sequence controls {len(cases)}",flush=True)
        pd.DataFrame(rows).to_csv(output/"eligibility.csv",index=False,lineterminator="\n")
        write_json(output/"diagnostics.json",traces);write_json(output/"sequence_cases.json",cases)
        summary=dict(source_records=len(all_rows),screened=len(rows),reference_verified=sum(r["reference_verified"] for r in rows),
            sequence_controls=len(cases),reasons=dict(Counter(r["reason"] for r in rows)),
            native_inference="not run by design",head_search="not run",confirmation_ready=False,
            biological_eligible_loci=None,usable_donors=None,
            missing_qc=["donor and biosample identity","per-replicate counts","individual genotype and phase",
                        "mapping-bias provenance","source overlap and homology","intervention-based power"],
            interpretation="sequence-only feasibility of single-SNV reference scenarios; not binding agreement or independent biological replication")
        write_json(output/"summary.json",summary)
        write_json(output/"inspection_ledger.json",dict(dataset="ADASTRA CTCF Mabel v6.1",role="exploratory only",
            source_records=len(all_rows),selected_membership_sha256=sha256_file(output/"membership.csv"),
            source_table_opened=True,outcomes_used=False,fresh_confirmation_membership=None,
            prior_study="results/variant_study/inspection_ledger.json records the earlier metadata-only inspection"))
        receipt["status"]="completed";print(json.dumps(summary,indent=2),flush=True)
    except Exception as exc:
        receipt.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        receipt.update(elapsed_seconds=time.time()-started,source_changed_during_run=any(sha256_file(ROOT/n)!=h for n,h in hashes.items()),
            artifacts={p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name not in ("execution.json","download.json")})
        write_json(output/"execution.json",receipt)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,default=ROOT/"data/adastra/ADASTRA.v.6.1.Mabel.zip")
    parser.add_argument("--output",type=Path,default=ROOT/"results/adastra_exploratory")
    parser.add_argument("--genome",type=Path,default=ROOT/"data/genomes/hg38.fa")
    parser.add_argument("--limit",type=int,default=4096)
    args=parser.parse_args();run(args.archive,args.output,args.genome,args.limit)
