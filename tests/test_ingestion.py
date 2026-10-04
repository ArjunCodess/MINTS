from pathlib import Path
from src.config import DataConfig

import pytest

from src.data_ingestion import (
    canonicalize_task_name,
    encode_output_filename,
    filter_encode_artifact_urls,
    is_encode_artifact_url,
    read_encode_urls,
    partition_digest,
)


def test_cached_partition_identity_includes_labels_and_tokenization():
    original=dict(sequence="ACGT",label=1,name="chr1:0-4",input_ids=[1,2],attention_mask=[1,1])
    assert partition_digest([original])!=partition_digest([{**original,"label":0}])
    assert partition_digest([original])!=partition_digest([{**original,"input_ids":[1,3]}])


def test_task_aliases_are_canonicalized() -> None:
    assert canonicalize_task_name("splice_sites_donor") == "splice_sites_donors"
    assert canonicalize_task_name("splice_sites_acceptor") == "splice_sites_acceptors"
    assert canonicalize_task_name("splice_sites_donors") == "splice_sites_donors"
    assert canonicalize_task_name("splice_sites_acceptors") == "splice_sites_acceptors"
    assert canonicalize_task_name("promoter_tata") == "promoter_tata"


def test_unknown_task_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        canonicalize_task_name("not_a_real_task")


def test_encode_url_filter_keeps_only_requested_artifacts() -> None:
    urls = [
        "https://www.encodeproject.org/metadata/?type=Experiment",
        "https://www.encodeproject.org/files/ENCFF680XUD/@@download/ENCFF680XUD.bigWig",
        "https://www.encodeproject.org/files/ENCFF827JRI/@@download/ENCFF827JRI.bed.gz",
        "https://www.encodeproject.org/files/ENCFF511URZ/@@download/ENCFF511URZ.bigBed",
        "https://www.encodeproject.org/files/ENCFF000ABC/@@download/ENCFF000ABC.txt",
    ]

    assert filter_encode_artifact_urls(urls)==[urls[2]]
    requested=DataConfig(encode_allowed_suffixes=(".bigWig", ".bed.gz", ".bigBed"))
    filtered = filter_encode_artifact_urls(urls,requested)

    assert len(filtered) == 3
    assert all(is_encode_artifact_url(url,requested) for url in filtered)
    assert encode_output_filename(filtered[0]) == "ENCFF680XUD.bigWig"


def test_read_encode_urls_strips_quotes_and_comments(tmp_path: Path) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "\n".join(
            [
                "# comment",
                '"https://www.encodeproject.org/files/ENCFF680XUD/@@download/ENCFF680XUD.bigWig"',
                "",
                "not-a-url",
            ]
        ),
        encoding="utf-8",
    )

    assert read_encode_urls(url_file) == [
        "https://www.encodeproject.org/files/ENCFF680XUD/@@download/ENCFF680XUD.bigWig"
    ]
