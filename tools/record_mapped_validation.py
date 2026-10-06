"""Record completed scientific and manuscript validation without inventing checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import shutil
import subprocess
import time
from src.utils import write_json,sha256_file
from tools.audit_mapped_variant import audit

ROOT=Path(__file__).resolve().parents[1]


def main():
    validation=ROOT/"results/mapped_variant/validation"
    cuda=json.loads((validation/"cuda-final.json").read_text())
    cpu=json.loads((validation/"cpu-final.json").read_text())
    assert cuda["status"]==cpu["status"]=="completed"
    for label in ("cuda-final","cpu-final"):
        assert "192 passed" in (validation/f"{label}_tests.log").read_text(encoding="utf-8",errors="replace")
    build_bytes=(validation/"manuscript_compile.log").read_bytes()
    build=json.loads(build_bytes.decode("utf-16" if build_bytes.startswith(b"\xff\xfe") else "utf-8-sig"))
    # The compiler reports its own completed subprocess and output path.
    write_json(validation/"manuscript_build.json",dict(compiler_result=build,
        pdf_sha256=sha256_file(ROOT/"paper/main.pdf"),compile_log_sha256=sha256_file(validation/"manuscript_compile.log"),
        source_sha256={p.relative_to(ROOT).as_posix():sha256_file(p) for p in (ROOT/"paper").glob("*.tex")},
        bibliography_sha256=sha256_file(ROOT/"paper/references.bib"),
        visual_inspection="new result text pages 10/11 and figure page 12 inspected at 1300-pixel rendering; no clipping",
        log_inspection="final TeX log has no undefined citation or overfull box warnings",pages=16))
    write_json(validation/"summary.json",dict(status="completed local validation; clean checkout pending",
        tests=dict(cuda=192,cpu=192),offline_audits="variant, archived ADASTRA and mapped study passed in both environments",
        raw_logits_audit=audit(raw=True),manuscript="compiled; primary citations verified; result text and figure visually inspected",
        scientific_endpoint="negative exploratory variant-minus-sham sequence sensitivity; head search and confirmation stopped",
        failures_preserved=["cuda.json","cpu.json","cuda-retry.json","cpu-retry.json"],
        manuscript_initial_failure="Bibliography changed after reference verification; resolved by official citation verification and report rebuild",
        cleanup=dict(status="not performed",action="recursive removal of two generated test directories",
            returned_reason="blocked by policy",authorization="user authorized; automatic policy rejection, not user refusal"),
        downloaded_archive=dict(bytes=942669336,md5="4a3672e8261c49c37d8d2297393b0069",
            sha256="172c9e9d922abd22bd6c3fe53c64fc4f9d1d7243f3d7daca45ce2099fd99dbf2",checks="MD5/size and SHA256 reverified during this task"),
        disk_free_bytes=shutil.disk_usage(ROOT).free,recorded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())))
    print("completed validations recorded")


if __name__=="__main__":main()
