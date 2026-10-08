import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.artifact
def test_public_paper_artifacts_have_complete_lineage():
    lineage = json.loads((ROOT/'paper/figures/lineage.json').read_text())
    for category in ('inputs','generators','outputs'):
        for name, expected in lineage[category].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    assert lineage['scientific_revision'] == 'cc09923652f0b361f73a36f382c4358dcf8b8100'


@pytest.mark.artifact
def test_saved_correspondence_certificate_matches_every_saved_query():
    certificate = json.loads((ROOT/'results/correspondence_audit/query_certificate.json').read_text())
    cases = json.loads((ROOT/'results/mapped_variant/cases.json').read_text())
    assert certificate['query_count'] == sum(len(c['queries']) for c in cases) == 438
    assert len(cases) == certificate['cases'] == 61
    for case, checked in zip(cases,certificate['cases_checked'],strict=True):
        assert case['variant_id'] == checked['variant_id']
        assert case['queries'] == checked['queries']
        assert checked['candidate_accounting_complete']
        assert checked['token_counts'] == [len(ids) for ids in case['ids']]
