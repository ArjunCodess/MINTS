"""Capture actual test and evidence-audit processes in each local environment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import platform
import shutil
import subprocess
import time
from src.utils import write_json,sha256_file

ROOT=Path(__file__).resolve().parents[1]


def run(label):
    output=ROOT/"results/mapped_variant/validation"
    output.mkdir(parents=True,exist_ok=True)
    receipt=output/f"{label}.json"
    if receipt.exists():raise FileExistsError("Preserve prior validation; choose a new label")
    free=shutil.disk_usage(ROOT).free
    if free<2*1024**3:raise OSError("Require 2 GiB spare disk before tests")
    (ROOT/".test-tmp").mkdir(exist_ok=True)
    commands=[("dependencies",[sys.executable,"-m","pip","check"]),
        ("compileall",[sys.executable,"-m","compileall","-q","src","tools","tests"]),
        ("variant_audit",[sys.executable,"tools/audit_variant_study.py"]),
        ("adastra_audit",[sys.executable,"tools/audit_adastra_feasibility.py"]),
        ("mapped_audit",[sys.executable,"tools/audit_mapped_variant.py"]),
        ("tests",[sys.executable,"-m","pytest","-q","--basetemp",f".test-tmp/mapped-{label}"])]
    values=dict(status="running",python=platform.python_version(),executable=sys.executable,disk_free_bytes=free,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),commands=[])
    write_json(receipt,values)
    for name,command in commands:
        start=time.time();log=output/f"{label}_{name}.log"
        with log.open("wb") as stream:
            process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        values["commands"].append(dict(name=name,command=command,exit_code=process.returncode,
            elapsed_seconds=time.time()-start,log_sha256=sha256_file(log)))
        write_json(receipt,values)
        print(f"{label}: {name} exit={process.returncode}",flush=True)
        if process.returncode:
            values["status"]="failed";write_json(receipt,values);raise RuntimeError(f"{name} failed, see {log}")
    values["status"]="completed";write_json(receipt,values)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--label",required=True)
    a=p.parse_args();run(a.label)
