"""Run cache-independent checks in a clean checkout, retaining exact process logs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import os
import shutil
import subprocess
import time
from src.utils import write_json,sha256_file

ROOT=Path(__file__).resolve().parents[1]


def main(checkout,label="clean"):
    checkout=checkout.resolve()
    assert checkout!=ROOT and (checkout/".git").is_dir()
    assert all(not(checkout/p).exists() for p in ("data/adastra","data/genomes","data/ctcf","data/hf_downstream"))
    output=ROOT/"results/mapped_variant/validation"
    receipt=output/f"{label}_checkout.json"
    if receipt.exists():raise FileExistsError("Preserve prior validation")
    if shutil.disk_usage(checkout).free<2*1024**3:raise OSError("Require 2 GiB spare disk")
    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HOME=str(checkout/".empty-hf-cache"))
    python=ROOT/".venv-cpu-check/Scripts/python.exe"
    values=dict(status="running",checkout=str(checkout),commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=checkout,text=True).strip(),
        scientific_caches_absent=True,empty_offline_hf_cache=env["HF_HOME"],disk_free_bytes=shutil.disk_usage(checkout).free,
        commands=[])
    write_json(receipt,values)
    commands=[(f"{label}_variant_audit",[str(python),"tools/audit_variant_study.py"]),
        (f"{label}_adastra_audit",[str(python),"tools/audit_adastra_feasibility.py"]),
        (f"{label}_mapped_audit",[str(python),"tools/audit_mapped_variant.py"]),
        (f"{label}_tests",[str(python),"-m","pytest","-q","-m","not artifact","--basetemp",".test-tmp"]),
        (f"{label}_report",[str(python),"tools/build_mapped_report.py"])]
    original=json.loads((checkout/"results/mapped_variant/report_manifest.json").read_text())
    for name,command in commands:
        log=output/f"{name}.log";start=time.time()
        with log.open("wb") as stream:
            proc=subprocess.run(command,cwd=checkout,env=env,stdout=stream,stderr=subprocess.STDOUT)
        values["commands"].append(dict(command=command,name=name,exit_code=proc.returncode,
            elapsed_seconds=time.time()-start,log_sha256=sha256_file(log)))
        write_json(receipt,values)
        print(f"{name} exit={proc.returncode}",flush=True)
        if proc.returncode:
            values["status"]="failed";write_json(receipt,values);raise RuntimeError(f"{name} failed")
    rebuilt=json.loads((checkout/"results/mapped_variant/report_manifest.json").read_text())
    values["report_outputs_byte_identical"]=original["outputs"]==rebuilt["outputs"]
    assert values["report_outputs_byte_identical"]
    assert "182 passed, 10 deselected" in (output/f"{label}_tests.log").read_text(encoding="utf-8",errors="replace")
    values.update(status="completed",tests_passed=182,artifact_tests_deselected=10)
    write_json(receipt,values)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("checkout",type=Path)
    parser.add_argument("--label",default="clean")
    args=parser.parse_args();main(args.checkout,args.label)
