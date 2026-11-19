"""FastAPI application entry point: `uvicorn app.main:app`."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app import __version__
from app.api.routes import router as api_router
from app.api.ui import router as ui_router
from app.config import get_settings
from app.db import init_db
from app.evaluation.runner import mark_interrupted_experiments

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Create data folders and tables; mark experiments left over from a crashed process."""
    get_settings().ensure_dirs()
    init_db()
    mark_interrupted_experiments()
    yield


app = FastAPI(
    title="HalluGuard",
    version=__version__,
    description="Layered hallucination detection and mitigation workflow for LLMs, with batch experiments.",
    lifespan=lifespan,
)
app.include_router(ui_router)
app.include_router(api_router)
