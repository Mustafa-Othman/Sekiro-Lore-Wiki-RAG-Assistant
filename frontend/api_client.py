"""Thin wrapper around the backend's query endpoints.

The backend base URL is read from the `API_BASE_URL` environment variable and is
never hard-coded, so pointing the UI at a deployed backend requires no code
change. See `.env.example`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env from the frontend directory, regardless of launch directory.
load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 300  # a local LLM on CPU can be slow; be generous


class ApiClientError(RuntimeError):
    """Raised with a message that is safe and useful to show to a user."""


@dataclass
class Detection:
    boss: str
    confidence: float
    used_for_retrieval: bool


@dataclass
class Answer:
    answer: str
    sources: list[str] = field(default_factory=list)
    detection: Detection | None = None


def get_base_url() -> str:
    """Backend base URL, from the environment."""
    return os.getenv("API_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _parse(payload: dict) -> Answer:
    raw_detection = payload.get("detection")
    detection = (
        Detection(
            boss=raw_detection["boss"],
            confidence=raw_detection["confidence"],
            used_for_retrieval=raw_detection.get("used_for_retrieval", False),
        )
        if raw_detection
        else None
    )
    return Answer(
        answer=payload.get("answer", ""),
        sources=payload.get("sources", []),
        detection=detection,
    )


def _raise_for_status(response: requests.Response) -> None:
    """Translate HTTP errors into messages worth showing a user."""
    if response.ok:
        return

    detail = ""
    try:
        body = response.json()
        detail = body.get("detail", "") if isinstance(body, dict) else ""
        if isinstance(detail, list) and detail:
            # FastAPI validation errors arrive as a list of field errors.
            detail = "; ".join(str(item.get("msg", item)) for item in detail)
    except ValueError:
        detail = response.text[:200]

    if response.status_code == 422:
        raise ApiClientError(f"The backend rejected that request: {detail}")
    if response.status_code == 503:
        raise ApiClientError(
            detail
            or "The backend's vector store returned no results. Is the store populated?"
        )
    if response.status_code == 502:
        # The backend's own detail here is already written for a human, so
        # prefer it over a second, near-identical wrapper sentence.
        raise ApiClientError(
            detail
            or "The backend could not reach the LLM. Is Ollama running and is the "
               "model pulled?"
        )
    raise ApiClientError(f"Backend error {response.status_code}: {detail}")


def ask(question: str, image_bytes: bytes | None = None,
        image_name: str = "screenshot.png") -> Answer:
    """Ask a question, optionally attaching a screenshot for boss detection.

    When `image_bytes` is given this calls `/query-image`, which biases retrieval
    toward the boss detected in the image (if confidence clears the backend's
    threshold). Otherwise it calls the plain text endpoint.
    """
    base = get_base_url()
    try:
        if image_bytes is None:
            response = requests.post(
                f"{base}/query", json={"question": question}, timeout=REQUEST_TIMEOUT
            )
        else:
            response = requests.post(
                f"{base}/query-image",
                data={"question": question},
                files={"image": (image_name, image_bytes, "image/png")},
                timeout=REQUEST_TIMEOUT,
            )
    except requests.exceptions.ConnectionError as exc:
        raise ApiClientError(
            f"Could not reach the backend at {base}. Is it running? "
            f"(uvicorn app.main:app --reload)"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ApiClientError(
            "The backend took too long to respond. A local LLM on CPU can be slow "
            "-- try a shorter question or a smaller model."
        ) from exc

    _raise_for_status(response)
    try:
        return _parse(response.json())
    except ValueError as exc:
        raise ApiClientError("The backend returned a response that was not valid JSON.") from exc


def health() -> dict:
    """Backend health payload, for the sidebar's status display."""
    base = get_base_url()
    try:
        response = requests.get(f"{base}/health", timeout=10)
    except requests.exceptions.RequestException as exc:
        raise ApiClientError(f"Backend unreachable at {base}.") from exc
    _raise_for_status(response)
    return response.json()
