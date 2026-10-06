"""Report generation must preserve evidence and describe its actual inputs."""
import json
from pathlib import Path
import shutil

import pytest

from tools import build_variant_artifacts as builder

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def report_tree(tmp_path,monkeypatch):
    study=tmp_path/"results/custom study"
    pilot=tmp_path/"results/custom pilot"
    shutil.copytree(ROOT/"results/variant_study",study)
    pilot.mkdir(parents=True)
    for name in ("summary.json","gate.json","eligibility.csv","eligibility_diagnostics.json","execution.json"):
        shutil.copyfile(ROOT/"results/variant_pilot"/name,pilot/name)
    for name in ("src/variant_audit.py","tools/audit_variant_study.py","docs/MINTS_v2_protocol.md"):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,path)
    monkeypatch.setattr(builder,"ROOT",tmp_path)
    monkeypatch.setattr(builder,"audit_study",lambda *a,**kw:dict(status="verified"))
    return tmp_path,study,pilot


@pytest.mark.parametrize("target",[
    "results/custom pilot/protocol.json",
    "results/custom study/report.md",
    "docs/MINTS_v2_protocol.md",
    "README.md",
    "src/variant_assay.py",
    "../outside.md",
])
def test_report_cannot_overwrite_evidence(report_tree,target):
    root,study,pilot=report_tree
    before={p.relative_to(root):p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):builder.build(study,pilot,root/target)
    assert before=={p.relative_to(root):p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_nonempty_result_is_rejected_without_output_writes(report_tree):
    root,study,pilot=report_tree
    summary=json.loads((pilot/"summary.json").read_text());summary["retained"]=1
    (pilot/"summary.json").write_text(json.dumps(summary))
    before={p.relative_to(root):p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError,match="nonempty"):builder.build(study,pilot,root/"docs/new.md")
    assert before=={p.relative_to(root):p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_custom_report_records_quoted_paths_and_relative_links(report_tree):
    root,study,pilot=report_tree
    report=root/"exports/reader's report.md"
    builder.build(study,pilot,report)
    text=report.read_text()
    assert "--study 'results/custom study' --pilot 'results/custom pilot'" in text
    assert "--report 'exports/reader''s report.md'" in text
    assert "../docs/variant_verification.md#fresh-output-reproduction" in text
    assert "../docs/MINTS_v2_protocol.md" in text
    receipt=json.loads((study/"report_manifest.json").read_text())
    assert receipt["outputs"][report.relative_to(root).as_posix()]==builder.sha256_file(report)
    assert len(receipt["outputs"])==4
    for name in ("exclusion_counts.csv","sham_constraint_counts.csv","token_boundary_diagnostics.csv"):
        assert b"\r\n" not in (study/name).read_bytes()
    first={name:(root/name).read_bytes() for name in receipt["outputs"]}
    builder.build(study,pilot,report)
    assert first=={name:(root/name).read_bytes() for name in receipt["outputs"]}
