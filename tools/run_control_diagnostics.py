"""Freeze and execute an expanded exploratory sequence-only diagnostic."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter
import gzip
import importlib.metadata
import json
import platform
import shutil
import time

import pandas as pd
from pyfaidx import Fasta
from transformers import AutoTokenizer

from src.adastra_candidates import allele_window, select_candidates
from src.config import DEFAULT_CONFIG
from src.control_diagnostics import diagnose_window
from src.controlled_edits import MotifDefinition
from src.motif_scoring import load_jaspar_ctcf_motif, motif_pssm, find_jaspar_matrix_path
from src.utils import sha256_file, write_json
from src.variant_protocol import VariantProtocol

ROOT = Path(__file__).resolve().parents[1]


def run(output, limit):
    if output.exists():
        raise FileExistsError("A fresh output directory is required")
    free = shutil.disk_usage(ROOT).free
    if free < 2 * 1024**3:
        raise OSError("At least 2 GiB free required")
    output.mkdir(parents=True)
    source = ROOT / "results/adastra_exploratory/candidates.csv.gz"
    sources = ["src/control_diagnostics.py", "tools/run_control_diagnostics.py", "src/adastra_candidates.py",
        "src/assay_alignment.py", "src/controlled_edits.py", "src/motif_scoring.py", "src/native_followup.py",
        "src/variant_protocol.py", "src/config.py", "src/utils.py"]
    hashes = {n: sha256_file(ROOT/n) for n in sources}
    protocol = dict(version="control-diagnostics-v1", role="exploratory method development; no model scoring",
        prior_inspection="complete ADASTRA source already exploratory; original 4096 inspected",
        source_sha256=sha256_file(source), source_code=hashes, seed=1731, limit=limit,
        membership="first limit coordinate-and-allele hashes; original first 4096 labelled separately; expansion disjoint",
        windows="204 bp hg38; SNV index 102, forward reference orientation",
        motif="MA0139.1, existing 0.8 support rule; both strands",
        predicates="independent on exact-substitution and trinucleotide candidate risk set; count all sites before risk set",
        alternatives=["original", "local_only", "local_no_width", "local_geometry32", "local_geometry64"],
        interpretation="descriptive intersections and retention, no significance or biological association; no new confirmation",
        reference_constraints=vars(VariantProtocol()), disk_free_bytes=free)
    write_json(output/"protocol.json", protocol)
    started = time.time()
    receipt = dict(status="running", started_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        protocol_sha256=sha256_file(output/"protocol.json"), source_sha256=hashes,
        python=platform.python_version(), packages={n:importlib.metadata.version(n) for n in ("transformers","tokenizers","numpy","pandas","pyfaidx")})
    write_json(output/"execution.json",receipt)
    try:
        table = pd.read_csv(source)
        selected = select_candidates(table, limit)
        selected["cohort"] = ["original4096" if i < 4096 else "expansion" for i in range(len(selected))]
        selected.to_csv(output/"membership.csv",index=False,lineterminator="\n")
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CONFIG.model.model_name,
            revision=DEFAULT_CONFIG.model.revision,trust_remote_code=True,local_files_only=True)
        motif = MotifDefinition("CTCF",pssm=motif_pssm(load_jaspar_ctcf_motif()),fraction=.8)
        rows, counts = [], Counter()
        candidate_path = output/"candidate_predicates.csv.gz"
        with Fasta(str(ROOT/"data/genomes/hg38.fa"),as_raw=True,sequence_always_upper=True,rebuild=False) as genome, \
                gzip.open(candidate_path,"wt",encoding="utf-8",newline="") as stream:
            header = True
            for number, row in enumerate(selected.to_dict("records"),1):
                window,reason = allele_window(genome,row)
                values,candidates = diagnose_window(tokenizer,window,motif,VariantProtocol()) if window else ({},[])
                rows.append(dict(variant_id=row["variant_id"],cohort=row["cohort"],reference_verified=window is not None,
                                 reference_reason=reason,**values))
                if candidates:
                    pd.DataFrame([dict(variant_id=row["variant_id"],cohort=row["cohort"],**c) for c in candidates]).to_csv(
                        stream,index=False,header=header,lineterminator="\n")
                    header=False
                if number % 256 == 0:
                    pd.DataFrame(rows).to_csv(output/"cases.csv",index=False,lineterminator="\n")
                    print(f"diagnosed {number}/{limit}",flush=True)
        frame=pd.DataFrame(rows)
        frame.to_csv(output/"cases.csv",index=False,lineterminator="\n")
        keys=["reference_verified","full_allele_offsets","reference_motif","query_correspondence",
              "original","local_only","local_no_width","local_geometry32","local_geometry64"]
        write_json(output/"summary.json",dict(source_records=len(table),cohorts={cohort:dict(records=len(group),
            **{k:int(group[k].fillna(False).sum()) for k in keys}) for cohort,group in frame.groupby("cohort")},
            candidate_risk_set="same substitution and exact trinucleotide, independent predicates; not sequential rejection counts"))
        write_json(output/"inspection_ledger.json",dict(role="exploratory",outcomes_used=False,
            membership_sha256=sha256_file(output/"membership.csv"),native_scores_opened=False))
        receipt["status"]="completed"
    except Exception as exc:
        receipt.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt.update(elapsed_seconds=time.time()-started,source_changed_during_run=any(sha256_file(ROOT/n)!=h for n,h in hashes.items()),
            artifacts={p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name!="execution.json"})
        write_json(output/"execution.json",receipt)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/control_diagnostics")
    parser.add_argument("--limit",type=int,default=12288)
    args=parser.parse_args()
    run(args.output,args.limit)
