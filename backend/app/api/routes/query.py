"""Query endpoints: `GET /health`, `POST /query`, `POST /query-image`."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from app.schemas.query import DetectionResult, QueryRequest, QueryResponse
from app.services.generation import resolve_boss_filter

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


def _services(request: Request):
    """Pull the startup-loaded services off application state."""
    return request.app.state.retrieval, request.app.state.generation, request.app.state.detection


@router.get("/health", summary="Service status")
def health(request: Request) -> dict:
    """Report readiness of each startup-loaded dependency.

    Returns 200 even when a dependency is degraded, so the endpoint stays useful
    for diagnosis rather than mirroring the failure. `status` is "ok" only when
    everything needed to answer a Core Track question is up.
    """
    retrieval, generation, detection = _services(request)
    settings = request.app.state.settings

    model_ok, model_detail = generation.is_model_available()

    return {
        "status": "ok" if (retrieval.is_loaded and model_ok) else "degraded",
        "vector_store": {
            "loaded": retrieval.is_loaded,
            "chunks": retrieval.chunk_count,
            "collection": settings.vector_store_collection,
            "boss_classes": retrieval.boss_classes,
        },
        "llm": {
            "available": model_ok,
            "model": settings.ollama_model,
            "host": settings.ollama_host,
            "detail": model_detail,
        },
        "detection": {
            "enabled": detection.is_enabled,
            "confidence_threshold": settings.yolo_confidence_threshold,
        },
    }


def _answer(question: str, retrieval, generation, boss: str | None, detected_boss: str | None = None) -> QueryResponse:
    """Shared retrieve -> prompt -> generate path for both endpoints."""
    chunks = retrieval.retrieve(question, boss=boss)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store returned no results; check that the store is populated.",
        )

    try:
        answer = generation.generate(question, chunks, detected_boss=detected_boss)
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Could not reach the LLM at {generation._settings.ollama_host}. "
                f"Is Ollama running and is the model pulled? ({exc.__class__.__name__})"
            ),
        ) from exc

    # De-duplicate while preserving retrieval order: several chunks often come
    # from the same page, and a citation list repeating one page adds no value.
    sources: list[str] = []
    for chunk in chunks:
        if chunk.label not in sources:
            sources.append(chunk.label)

    return QueryResponse(answer=answer, sources=sources)


@router.post("/query", response_model=QueryResponse, summary="Ask a question")
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    """Answer a question from the Sekiro wiki corpus, with cited sources."""
    retrieval, generation, _ = _services(request)
    logger.info("Query: %s", payload.question)
    return _answer(payload.question, retrieval, generation, boss=None, detected_boss=None)


@router.post(
    "/query-image",
    response_model=QueryResponse,
    summary="Ask a question about an uploaded screenshot (Extended Track)",
)
async def query_with_image(
    request: Request,
    question: str = Form(..., min_length=1, max_length=1000),
    image: UploadFile = File(...),
) -> QueryResponse:
    """Answer a question, biasing retrieval toward a boss detected in the image.

    Implements the Section 5.5.8 rule via `resolve_boss_filter`: the detection
    only constrains retrieval when it is at or above
    `YOLO_CONFIDENCE_THRESHOLD`. Below that -- or when detection is disabled --
    this behaves exactly like `POST /query`, so an uncertain detection never
    blocks a normal answer. The detection is always reported back for
    transparency.
    """
    retrieval, generation, detection = _services(request)
    settings = request.app.state.settings

    if not question.strip():
        raise HTTPException(status_code=422, detail="question must not be blank")

    image_bytes = await image.read()
    logger.info("Image query (%d bytes): %s", len(image_bytes), question)

    detection_result: DetectionResult | None = None
    raw = detection.detect(image_bytes) if detection.is_enabled else None
    detected_boss_name = None

    if raw is not None:
        boss_class, confidence = raw
        filter_boss = resolve_boss_filter(
            DetectionResult(
                boss=boss_class, confidence=confidence, used_for_retrieval=False
            ),
            settings.yolo_confidence_threshold,
        )
        detection_result = DetectionResult(
            boss=boss_class,
            confidence=confidence,
            used_for_retrieval=filter_boss is not None,
        )
        detected_boss_name = boss_class
        logger.info(
            "Detected %s (%.2f) -- retrieval filter %s",
            boss_class, confidence, "applied" if filter_boss else "skipped",
        )
    else:
        filter_boss = None
        logger.info("No confident detection -- standard similarity search")

    response = _answer(question.strip(), retrieval, generation, boss=filter_boss, detected_boss=detected_boss_name)
    response.detection = detection_result
    return response
