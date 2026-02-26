from typing import List

from pydantic import BaseModel, Field


class Fact(BaseModel):
    """An atomic unit of knowledge."""

    content: str = Field(description="The factual statement")
    source_url: str = Field(description="Where this fact came from")
    confidence_score: int = Field(description="1-100 confidence score")
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
