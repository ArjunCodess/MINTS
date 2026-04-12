from pathlib import Path

import pytest

from src.data_ingestion import (
    canonicalize_task_name,
    encode_output_filename,
    filter_encode_artifact_urls,
    is_encode_artifact_url,
    read_encode_urls,
)


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

    filtered = filter_encode_artifact_urls(urls)

    assert len(filtered) == 3
    assert all(is_encode_artifact_url(url) for url in filtered)
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
