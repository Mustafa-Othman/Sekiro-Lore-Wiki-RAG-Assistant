"""Request/response models for the query API."""

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """A question asked against the Sekiro wiki corpus."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to answer from the retrieved wiki context.",
        examples=["What does the Mortal Blade do?"],
    )

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """Reject whitespace-only questions, not just empty ones.

        `min_length=1` alone would let `"   "` through and send a meaningless
        query to the retriever.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class DetectionResult(BaseModel):
    """Outcome of running the YOLO detector on an uploaded screenshot."""

    boss: str = Field(..., description="Detected class name, matching the `boss` metadata field.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    # False when confidence fell below YOLO_CONFIDENCE_THRESHOLD. The detection is
    # still reported for transparency, but retrieval was NOT filtered by it.
    used_for_retrieval: bool = Field(
        ...,
        description="Whether this detection was confident enough to filter retrieval.",
    )


class QueryResponse(BaseModel):
    """A grounded answer plus the wiki sources it was built from."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    detection: DetectionResult | None = Field(
        default=None,
        description="Present only when an image was submitted (Extended Track).",
    )
