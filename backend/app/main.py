"""FastAPI application entrypoint.

Startup lifecycle
-----------------
The vector store, the LLM client and the YOLO detector are all loaded **once**,
in the `lifespan` context manager, and stashed on `app.state`. Handlers read them
from there. Nothing heavy is constructed per request -- rebuilding the store or
re-loading the embedding model on every call would be both slow and wrong.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import query as query_routes
from app.core.config import get_settings
from app.services.detection import DetectionService
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService
from app.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every heavyweight dependency once, before serving traffic."""
    configure_logging()
    settings = get_settings()
    app.state.settings = settings

    logger.info("Starting up: loading vector store and models")

    retrieval = RetrievalService(settings)
    retrieval.load()                      # raises if the store is missing
    app.state.retrieval = retrieval

    generation = GenerationService(settings)
    generation.load()                     # non-fatal if Ollama is down
    app.state.generation = generation

    detection = DetectionService(settings)
    detection.load()                      # non-fatal if weights/ultralytics absent
    app.state.detection = detection

    model_ok, detail = generation.is_model_available()
    logger.info("LLM: %s", detail if not model_ok else f"{settings.ollama_model} ready")
    logger.info("Detection enabled: %s", detection.is_enabled)
    logger.info("Startup complete -- %d chunks indexed", retrieval.chunk_count)

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Sekiro Lore & Wiki RAG Assistant",
    description=(
        "Answers questions about Sekiro: Shadows Die Twice from a retrieved wiki "
        "corpus, with cited sources. Optionally accepts a gameplay screenshot and "
        "biases retrieval toward the boss detected in it."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

_settings = get_settings()

# The frontend's origin comes from config -- never hard-coded, so pointing the UI
# at a different host needs no code change.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_routes.router)
