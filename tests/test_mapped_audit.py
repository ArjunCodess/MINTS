from pathlib import Path
import shutil
import pytest
from tools.audit_mapped_variant import audit

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.artifact
def test_mapped_evidence_offline():
    assert audit()["native_gate"]=="failed"


@pytest.mark.artifact
def test_changed_mapped_gate_is_rejected(tmp_path):
    for name in ("mapped_variant","control_diagnostics"):
        shutil.copytree(ROOT/"results"/name,tmp_path/"results"/name)
    path=tmp_path/"results/mapped_variant/summary.json"
    text=path.read_text().replace('"native_sensitivity": false','"native_sensitivity": true')
    path.write_text(text)
    with pytest.raises(AssertionError):audit(tmp_path)
