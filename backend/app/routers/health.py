from fastapi import APIRouter

from app.core.config import settings
from app.services import classifier as clf

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    c = clf.classifier
    return {
        "status": "ok",
        "model": c.model_name if c else None,
        "models": settings.model_names,
        "ensemble": c.is_ensemble if c else len(settings.model_names) > 1,
        "model_loaded": c is not None,
        "device": c.device if c else None,
    }
