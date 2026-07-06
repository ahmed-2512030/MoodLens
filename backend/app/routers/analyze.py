from fastapi import APIRouter

from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.classifier import get_classifier

router = APIRouter(tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    result = get_classifier().classify([req.text])[0]
    return AnalyzeResponse(text=req.text, **result)
