import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.routers import analyze, health, report, upload
from app.services.classifier import load_classifier

logger = logging.getLogger("moodlens")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once, before serving traffic.
    load_classifier()
    yield


app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Always answer with JSON, even on unexpected failures. -------------------
# Without these, an unhandled error returns Starlette's plain-text "Internal
# Server Error", and the browser's res.json() then throws "invalid JSON",
# masking the real cause. These guarantee a JSON {"detail": ...} body every time.
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # Flatten pydantic's error list into one readable sentence for the UI.
    detail = "; ".join(
        f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}"
        for e in exc.errors()
    ) or "Invalid request."
    return JSONResponse(status_code=422, content={"detail": detail})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal error while processing the request."},
    )

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(upload.router)
app.include_router(report.router)
