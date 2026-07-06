from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import analyze, health, report, upload
from app.services.classifier import load_classifier


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

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(upload.router)
app.include_router(report.router)
