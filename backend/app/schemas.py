from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    # Long text is chunked past the 512-token limit (see classifier.classify_document),
    # so the cap is a DoS guard, not an analysis limit. ~50k chars ≈ the 40-chunk ceiling.
    text: str = Field(..., min_length=1, max_length=50000)


class FineGrained(BaseModel):
    label: str
    score: float


class EmotionResult(BaseModel):
    dominant: str
    emotions: dict[str, float]
    fine_grained: list[FineGrained]


class ArcPoint(BaseModel):
    """One chunk's Ekman vector (0-100 scale). Shaped to match the frontend
    TrendPoint so the per-document arc drops straight into the trend chart."""
    bin: str
    index: int
    joy: float
    anger: float
    sadness: float
    fear: float
    surprise: float
    disgust: float
    neutral: float


class AnalyzeResponse(EmotionResult):
    text: str
    chunk_count: int = 1
    arc: list[ArcPoint] | None = None  # per-chunk trajectory; null when single-chunk
    truncated_chunks: bool = False


class BatchItem(EmotionResult):
    text: str


class EmotionSummary(BaseModel):
    """Aggregate counts/averages across a batch, for the dashboard charts."""
    count: int
    distribution: dict[str, int]          # dominant-label counts (pie chart)
    average_scores: dict[str, float]      # mean per-emotion score


class BatchResponse(BaseModel):
    summary: EmotionSummary
    items: list[BatchItem]
