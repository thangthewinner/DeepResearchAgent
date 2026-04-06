from src.models.schemas import Fact
from src.utils.evidence import (
    build_source_locator,
    extract_tool_result_payload,
    format_evidence_record,
    format_fact_for_writer,
    sort_facts_by_strength,
)


def test_extract_tool_result_payload_reads_text_and_structured_metadata() -> None:
    payload = (
        [{"type": "text", "text": "Rendered article body"}],
        {
            "structured_content": {
                "url": "https://example.com/article",
                "title": "Example Article",
                "page": 3,
            }
        },
    )

    content, metadata = extract_tool_result_payload(payload)

    assert content == "Rendered article body"
    assert metadata["title"] == "Example Article"
    assert metadata["page"] == 3


def test_build_source_locator_combines_supported_fields() -> None:
    locator = build_source_locator(
        {"page": 12, "doi": "10.1000/test", "section": "Results"}
    )

    assert locator == "page: 12; section: Results; DOI: 10.1000/test"


def test_format_evidence_record_includes_provenance_metadata() -> None:
    record = format_evidence_record(
        tool_name="parse_document",
        source_type="pdf",
        evidence_type="document_parse",
        content="Important paragraph",
        source_url="https://example.com/report.pdf",
        source_title="Annual Report",
        source_domain="example.com",
        source_locator="page: 14",
        published_at="2026-03-01",
        authors=["Jane Doe"],
    )

    assert "<EVIDENCE_RECORD>" in record
    assert "source_type: pdf" in record
    assert "source_locator: page: 14" in record
    assert "authors: Jane Doe" in record
    assert "<CONTENT>\nImportant paragraph" in record


def test_sort_facts_by_strength_prioritizes_stronger_evidence() -> None:
    snippet_fact = Fact(
        content="Snippet fact",
        source_url="https://news.example.com",
        confidence_score=70,
        source_type="web",
        evidence_type="snippet",
    )
    pdf_fact = Fact(
        content="PDF fact",
        source_url="https://example.com/report.pdf",
        confidence_score=70,
        source_type="pdf",
        evidence_type="document_parse",
        source_locator="page: 8",
    )

    ordered = sort_facts_by_strength([snippet_fact, pdf_fact])

    assert ordered[0] == pdf_fact


def test_format_fact_for_writer_includes_locator_and_metadata() -> None:
    fact = Fact(
        content="Benchmark improved by 12%",
        source_url="https://arxiv.org/abs/1234.5678",
        confidence_score=91,
        source_type="scholarly",
        source_title="A Benchmark Paper",
        source_domain="arxiv.org",
        source_locator="arXiv: 1234.5678",
        evidence_type="abstract",
        published_at="2026",
        authors=["Alice", "Bob"],
    )

    formatted = format_fact_for_writer(fact)

    assert "source_type=scholarly" in formatted
    assert "locator=arXiv: 1234.5678" in formatted
    assert "authors=Alice, Bob" in formatted
