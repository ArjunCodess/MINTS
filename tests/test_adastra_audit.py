"""Downloaded cohort accounting must remain consistent after artifact edits."""
import json
from pathlib import Path
import shutil

import pytest

from src.utils import sha256_file
from tools.audit_adastra_feasibility import audit

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def cohort(tmp_path):
    output=tmp_path/"results/adastra_exploratory"
    shutil.copytree(ROOT/"results/adastra_exploratory",output)
    execution=json.loads((output/"execution.json").read_text())
    for name in set(execution["source_sha256"])|{"tools/build_adastra_report.py","docs/adastra_feasibility.md"}:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    return tmp_path,output


def test_complete_coordinate_denominator_verifies_offline(cohort):
    root,output=cohort
    assert audit(output,root)==dict(status="verified",source_records=512556,screened=4096,sequence_controls=0,confirmation_ready=False)


@pytest.mark.parametrize("name,edit,reason",[
    ("summary.json",lambda d:d.update(sequence_controls=1),"population"),
    ("summary.json",lambda d:d.update(confirmation_ready=True),"confirmation"),
    ("diagnostics.json",lambda d:d[0].update(candidate_positions_checked=1),"accounting"),
    ("source.json",lambda d:d.update(outcomes_used=True),"inference"),
    ("inspection_ledger.json",lambda d:d.update(fresh_confirmation_membership=[]),"ledger"),
])
def test_updated_hash_cannot_hide_inconsistent_claims(cohort,name,edit,reason):
    root,output=cohort;path=output/name
    payload=json.loads(path.read_text());edit(payload);path.write_text(json.dumps(payload),encoding="utf-8")
    receipt=json.loads((output/"execution.json").read_text());receipt["artifacts"][name]=sha256_file(path)
    (output/"execution.json").write_text(json.dumps(receipt),encoding="utf-8")
    with pytest.raises(ValueError,match=reason):audit(output,root)
