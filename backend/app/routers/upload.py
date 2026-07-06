import io
import json

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.schemas import BatchResponse
from app.services.aggregate import summarize
from app.services.classifier import get_classifier

router = APIRouter(tags=["upload"])

TEXT_COLUMNS = ("text", "content", "message", "review", "comment", "body")


def _extract_texts(df: pd.DataFrame) -> list[str]:
    col = next((c for c in df.columns if c.lower() in TEXT_COLUMNS), None)
    if col is None:
        col = df.columns[0]  # fall back to the first column
    texts = df[col].dropna().astype(str).str.strip()
    return [t for t in texts if t]


@router.post("/upload", response_model=BatchResponse)
async def upload(file: UploadFile = File(...)) -> BatchResponse:
    raw = await file.read()
    name = (file.filename or "").lower()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw))
        elif name.endswith(".json"):
            data = json.loads(raw)
            df = pd.DataFrame(data if isinstance(data, list) else [data])
        else:
            raise HTTPException(400, "Only .csv or .json files are supported")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface parse errors to client
        raise HTTPException(400, f"Could not parse file: {exc}") from exc

    texts = _extract_texts(df)
    if not texts:
        raise HTTPException(400, "No text rows found in file")
    if len(texts) > settings.max_rows:
        raise HTTPException(
            413, f"Too many rows ({len(texts)}); limit is {settings.max_rows}"
        )

    results = get_classifier().classify(texts)
    items = [{"text": t, **r} for t, r in zip(texts, results)]
    return BatchResponse(summary=summarize(results), items=items)
