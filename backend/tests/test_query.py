"""Backend tests.

Design note: these run **real retrieval** against the persisted vector store, but
the LLM call is stubbed. That keeps the suite deterministic and independent of
whether Ollama is running, while still exercising the whole HTTP -> validation ->
retrieval -> response-schema path. Retrieval is the graded part; the LLM is an
external dependency whose output is not assertable anyway.

Because the real embedding model is loaded (once, module-scoped), this file takes
~2-3 minutes on first run. That is the cost of testing against the real index.

Run from the `backend/` directory so the default `./data/vector_store` path resolves:

    cd backend && pytest -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.services.generation import resolve_boss_filter

# --- Stub LLM -------------------------------------------------------------


def _fake_generate(question: str, chunks: list) -> str:
    """Stand-in for Ollama that still proves the context reached the prompt."""
    return f"[stub answer] used {len(chunks)} chunks for: {question}"


@pytest.fixture(scope="module")
def client():
    """App with lifespan run, so the vector store is genuinely loaded."""
    from app.main import app

    with TestClient(app) as test_client:
        # Swap only the network call; retrieval stays real.
        app.state.generation.generate = _fake_generate
        yield test_client


# --- Happy path -----------------------------------------------------------


def test_health_reports_loaded_store(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["vector_store"]["loaded"] is True
    assert body["vector_store"]["chunks"] > 0
    # Boss classes come from the notebook's Section 5.2 metadata tagging.
    assert body["vector_store"]["boss_classes"]


def test_query_returns_grounded_answer_and_sources(client):
    response = client.post("/query", json={"question": "What does the Mortal Blade do?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].strip()
    assert len(body["sources"]) > 0
    # Citations must name a real wiki page, not a chunk id.
    assert any("Mortal Blade" in source for source in body["sources"])
    # Core Track path: no image submitted, so there is no detection.
    assert body["detection"] is None


def test_query_retrieves_relevant_page_for_obscure_question(client):
    """Grounding sanity check: an obscure question must reach the right page."""
    response = client.post(
        "/query", json={"question": "Who is the Sculptor, and what is his backstory?"}
    )

    assert response.status_code == 200
    assert any("Sculptor" in s for s in response.json()["sources"])


# --- Invalid input (422) --------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},                                  # field missing entirely
        {"question": ""},                    # empty string
        {"question": "   "},                 # whitespace only, fails the validator
        {"question": "x" * 1001},            # over the length limit
    ],
    ids=["missing", "empty", "blank", "too-long"],
)
def test_query_rejects_invalid_input(client, payload):
    response = client.post("/query", json=payload)

    assert response.status_code == 422


# --- Extended Track: image endpoint --------------------------------------


def test_query_image_falls_back_to_plain_retrieval_when_detection_disabled(client):
    """With no YOLO weights configured, an image must not break the request.

    Detection is disabled, so this has to behave exactly like POST /query. The
    attached bytes are never decoded in that path, which is why a dummy payload
    is enough here.
    """
    response = client.post(
        "/query-image",
        data={"question": "What does the Mortal Blade do?"},
        files={"image": ("shot.png", b"not-a-real-png", "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].strip()
    assert body["detection"] is None  # nothing detected -> no filter applied


# --- Boss-filter decision logic (pure function, no app needed) ------------


class _Detection:
    """Minimal stand-in for the DetectionResult schema."""

    def __init__(self, boss: str, confidence: float) -> None:
        self.boss = boss
        self.confidence = confidence


def test_no_detection_means_no_filter():
    assert resolve_boss_filter(None, threshold=0.5) is None


def test_confident_detection_filters_retrieval():
    detection = _Detection("guardian_ape", 0.91)
    assert resolve_boss_filter(detection, threshold=0.5) == "guardian_ape"


def test_confidence_exactly_at_threshold_filters():
    """The spec says 'at or above' the threshold, so equality must filter."""
    detection = _Detection("owl", 0.5)
    assert resolve_boss_filter(detection, threshold=0.5) == "owl"


def test_low_confidence_detection_is_ignored():
    """A weak detection must never misdirect retrieval."""
    detection = _Detection("owl", 0.31)
    assert resolve_boss_filter(detection, threshold=0.5) is None
