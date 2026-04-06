from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

from ..models.schemas import Fact

_SOURCE_PRIORITY = {
    "official": 0,
    "pdf": 1,
    "scholarly": 2,
    "news": 3,
    "web": 4,
}

_EVIDENCE_PRIORITY = {
    "document_parse": 0,
    "full_text": 1,
    "extracted_content": 2,
    "abstract": 3,
    "snippet": 4,
}

_TEXT_KEYS = (
    "content",
    "text",
    "body",
    "markdown",
    "summary",
    "abstract",
    "excerpt",
    "result",
    "output",
    "answer",
)


def derive_source_domain(source_url: str) -> str:
    """Return the normalized domain for a source URL."""
    if not source_url:
        return ""

    parsed = urlparse(source_url)
    return parsed.netloc.lower()


def truncate_content(content: str, max_chars: int) -> str:
    """Trim long tool output while preserving readability."""
    normalized = content.strip()
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized

    return normalized[: max_chars - 3].rstrip() + "..."


def normalize_string_list(value: Any) -> list[str]:
    """Normalize an arbitrary value into a list of strings."""
    if value is None:
        return []

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return [part for part in parts if part]

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    return [str(value).strip()] if str(value).strip() else []


def normalize_tool_content(content: Any) -> str:
    """Convert tool output into readable text."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, tuple) and len(content) == 2:
        return normalize_tool_content(content[0])

    if isinstance(content, dict):
        for key in _TEXT_KEYS:
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(content, ensure_ascii=False, indent=2)

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue

            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    text = str(item.get("text", "")).strip()
                    if text:
                        parts.append(text)
                    continue

                if item_type == "image":
                    image_ref = item.get("url") or item.get("mime_type") or "image"
                    parts.append(f"[image: {image_ref}]")
                    continue

                if item_type == "file":
                    file_ref = item.get("url") or item.get("mime_type") or "file"
                    parts.append(f"[file: {file_ref}]")
                    continue

                parts.append(json.dumps(item, ensure_ascii=False, indent=2))
                continue

            parts.append(str(item).strip())

        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def extract_tool_result_payload(raw_result: Any) -> tuple[str, dict[str, Any]]:
    """Split a tool result into readable content and structured metadata."""
    content = raw_result
    artifact: dict[str, Any] = {}

    if isinstance(raw_result, tuple) and len(raw_result) == 2:
        content, artifact_candidate = raw_result
        if isinstance(artifact_candidate, dict):
            structured_content = artifact_candidate.get("structured_content")
            if isinstance(structured_content, dict):
                artifact = structured_content
            else:
                artifact = artifact_candidate

    return normalize_tool_content(content), artifact


def build_source_locator(metadata: dict[str, Any]) -> str:
    """Build a compact locator string from metadata fields."""
    locators: list[str] = []
    locator_keys = (
        ("page", "page"),
        ("page_number", "page"),
        ("pages", "pages"),
        ("page_range", "pages"),
        ("section", "section"),
        ("doi", "DOI"),
        ("arxiv_id", "arXiv"),
        ("document_id", "document_id"),
    )

    for key, label in locator_keys:
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            locators.append(f"{label}: {text}")

    return "; ".join(locators)


def _clean_metadata_value(value: str) -> str:
    return " ".join(value.split()).strip()


def _format_metadata_line(key: str, value: str) -> str:
    cleaned = _clean_metadata_value(value)
    return f"{key}: {cleaned if cleaned else 'unknown'}"


def format_evidence_record(
    *,
    tool_name: str,
    source_type: str,
    evidence_type: str,
    content: str,
    source_url: str = "",
    source_title: str = "",
    source_domain: str = "",
    source_locator: str = "",
    published_at: str = "",
    authors: list[str] | None = None,
) -> str:
    """Format tool output into a metadata-rich evidence record."""
    normalized_authors = ", ".join(normalize_string_list(authors))
    domain = source_domain or derive_source_domain(source_url)
    truncated_content = truncate_content(content, max_chars=20000)

    metadata_lines = [
        _format_metadata_line("tool_name", tool_name),
        _format_metadata_line("source_type", source_type),
        _format_metadata_line("evidence_type", evidence_type),
        _format_metadata_line("source_title", source_title),
        _format_metadata_line("source_url", source_url),
        _format_metadata_line("source_domain", domain),
        _format_metadata_line("source_locator", source_locator),
        _format_metadata_line("published_at", published_at),
        _format_metadata_line("authors", normalized_authors),
    ]

    return (
        "<EVIDENCE_RECORD>\n"
        + "\n".join(metadata_lines)
        + "\n</EVIDENCE_RECORD>\n"
        + "<CONTENT>\n"
        + truncated_content
        + "\n</CONTENT>"
    )


def sort_facts_by_strength(facts: list[Fact]) -> list[Fact]:
    """Sort stronger evidence before weaker evidence for downstream writing."""
    return sorted(
        facts,
        key=lambda fact: (
            _SOURCE_PRIORITY.get(fact.source_type, 99),
            _EVIDENCE_PRIORITY.get(fact.evidence_type, 99),
            -fact.confidence_score,
            fact.source_url,
            fact.content,
        ),
    )


def format_fact_for_writer(fact: Fact) -> str:
    """Render a fact with provenance for report synthesis prompts."""
    parts = [f"- [{fact.confidence_score}%] {fact.content}"]

    if fact.source_type:
        parts.append(f"source_type={fact.source_type}")
    if fact.evidence_type:
        parts.append(f"evidence_type={fact.evidence_type}")
    if fact.source_title:
        parts.append(f"title={fact.source_title}")
    if fact.source_domain:
        parts.append(f"domain={fact.source_domain}")
    if fact.source_locator:
        parts.append(f"locator={fact.source_locator}")
    if fact.published_at:
        parts.append(f"published_at={fact.published_at}")
    if fact.authors:
        parts.append(f"authors={', '.join(fact.authors)}")
    parts.append(f"url={fact.source_url}")

    return " | ".join(parts)
