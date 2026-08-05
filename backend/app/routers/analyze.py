from fastapi import APIRouter

from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.classifier import get_classifier

router = APIRouter(tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    # classify_document chunks long text past the 512-token limit and returns a
    # per-chunk arc; short text stays a single chunk (identical to classify()).
    result = get_classifier().classify_document(req.text)
    return AnalyzeResponse(text=req.text, **result)
