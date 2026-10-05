"""Read-only offline integrity check, suitable for a clean checkout and CI."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import subprocess

from src.variant_audit import audit_study


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--study",type=Path)
    parser.add_argument("--pilot",type=Path)
    args=parser.parse_args()
    try:
        result=audit_study(args.root,args.study,args.pilot)
    except (ValueError,KeyError,OSError,TypeError,subprocess.CalledProcessError) as exc:
        print(json.dumps(dict(status="failed",error=str(exc)),indent=2));sys.exit(1)
    print(json.dumps(result,indent=2))
