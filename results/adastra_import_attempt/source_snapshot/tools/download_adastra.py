"""Download the pinned public release without loading or filtering outcomes."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import os
import time
import zipfile

import requests

from src.utils import write_json,sha256_file

ROOT=Path(__file__).resolve().parents[1]
URL="https://zenodo.org/api/records/14174114/files/ADASTRA.v.6.1.Mabel.zip/content"
SIZE=942669336
MD5="4a3672e8261c49c37d8d2297393b0069"


def verify_archive(path,size=SIZE,md5=MD5):
    value=hashlib.md5()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):value.update(chunk)
    if Path(path).stat().st_size!=size or value.hexdigest()!=md5:
        raise ValueError("Downloaded archive does not match the publisher's size and checksum")


def download(output,receipt):
    output,receipt=Path(output),Path(receipt)
    output.parent.mkdir(parents=True,exist_ok=True)
    started=time.time()
    record=dict(url=URL,release="Mabel v6.1",record="14174114",expected_bytes=SIZE,publisher_md5=MD5,
                archive_path=output.as_posix(),status="running",role="exploratory only; no confirmation freshness assigned")
    partial=output.with_suffix(output.suffix+".part")
    owns_partial=False
    try:
        if not output.exists():
            with requests.get(URL,stream=True,timeout=(20,60)) as response:
                response.raise_for_status()
                record.update(resolved_url=response.url,http_status=response.status_code)
                with partial.open("xb") as stream:
                    owns_partial=True
                    count=0;last=0
                    for chunk in response.iter_content(1024*1024):
                        stream.write(chunk);count+=len(chunk)
                        if count>SIZE:raise ValueError("Archive exceeds the pinned release size")
                        if count-last>=64*1024*1024:
                            print(f"downloaded {count:,} / {SIZE:,} bytes",flush=True);last=count
            verify_archive(partial)
            os.replace(partial,output)
        verify_archive(output)
        with zipfile.ZipFile(output) as archive:
            names=[dict(name=i.filename,bytes=i.file_size,compressed_bytes=i.compress_size)
                   for i in archive.infolist() if "CTCF" in i.filename.upper()]
        record.update(status="completed",bytes=output.stat().st_size,sha256=sha256_file(output),ctcf_members=names)
        return record
    except Exception as exc:
        record.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        if owns_partial and partial.exists():partial.unlink()
        raise
    finally:
        record["elapsed_seconds"]=time.time()-started
        write_json(receipt,record)
        print(json.dumps(record,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"data/adastra/ADASTRA.v.6.1.Mabel.zip")
    parser.add_argument("--receipt",type=Path,default=ROOT/"results/adastra_exploratory/download.json")
    args=parser.parse_args();download(args.output,args.receipt)
