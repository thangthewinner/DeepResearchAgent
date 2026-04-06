from typing import List

from pydantic import BaseModel, Field


class Fact(BaseModel):
    """An atomic unit of knowledge with provenance metadata."""

    content: str = Field(description="The factual statement")
    source_url: str = Field(description="Where this fact came from")
    confidence_score: int = Field(description="1-100 confidence score")
    source_type: str = Field(
        default="web",
        description="Source category such as web, pdf, scholarly, official, or news",
    )
    source_title: str = Field(default="", description="Readable title of the source")
    source_domain: str = Field(default="", description="Normalized source domain")
    source_locator: str = Field(
        default="",
        description="Locator such as page, page range, DOI, section, or arXiv identifier",
    )
    evidence_type: str = Field(
        default="snippet",
        description="Evidence strength such as snippet, extracted_content, document_parse, abstract, or full_text",
    )
    published_at: str = Field(
        default="", description="Publication date or year when available"
    )
    authors: List[str] = Field(default_factory=list, description="Source authors")
    is_disputed: bool = Field(
        default=False, description="If this fact conflicts with others"
    )


class Critique(BaseModel):
    """Adversarial feedback model."""

    author: str
    concern: str
    severity: int
    addressed: bool = Field(default=False, description="Has the supervisor fixed this?")


class QualityMetric(BaseModel):
    """Quality snapshot at iteration."""

    score: float
    feedback: str
    iteration: int


class Summary(BaseModel):
    """Webpage summary schema."""

    summary: str = Field(description="Concise summary")
    key_excerpts: str = Field(description="Important quotes")


class ClarifyWithUser(BaseModel):
    """User clarification decision."""

    need_clarification: bool
    question: str
    verification: str


class ResearchQuestion(BaseModel):
    """Research brief schema."""

    research_brief: str


class DraftReport(BaseModel):
    """Draft report schema."""

    draft_report: str


class EvaluationResult(BaseModel):
    """Quality evaluation result."""

    comprehensiveness_score: int = Field(description="0-10 score")
    accuracy_score: int = Field(description="0-10 score")
    coherence_score: int = Field(description="0-10 score")
    specific_critique: str


class FactExtraction(BaseModel):
    """Context pruning output."""

    new_facts: List[Fact]
