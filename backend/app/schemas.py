from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class FineGrained(BaseModel):
    label: str
    score: float


class EmotionResult(BaseModel):
    dominant: str
    emotions: dict[str, float]
    fine_grained: list[FineGrained]


class AnalyzeResponse(EmotionResult):
    text: str


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
