from fastapi import APIRouter

from app.core.config import settings
from app.services import classifier as clf

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": settings.model_name,
        "model_loaded": clf.classifier is not None,
        "device": clf.classifier.device if clf.classifier else None,
    }
