"""Answer generation: build the grounded prompt, call Ollama, return an answer.

Grounding policy
----------------
The assistant must answer from *retrieved context*, never from the LLM's own
memorised Sekiro knowledge -- that is the most heavily penalised failure mode in
this assignment. The system prompt therefore restricts the model to the supplied
passages and requires a fixed refusal string when they are insufficient, which is
much easier to detect than a vague non-answer.

Boss-filter decision logic (Extended Track, Section 5.5.8)
----------------------------------------------------------
When a request carries an image, the detector's output decides how retrieval runs.
This is the exact rule, kept here so it is traceable rather than implicit:

1. No image supplied
   -> standard similarity search on the typed question alone. Identical to the
      Core Track pipeline.

2. Image supplied, detector confidence **below** ``YOLO_CONFIDENCE_THRESHOLD``
   -> ignore the detection entirely and run standard similarity search. The
      detection is still returned in the response (with
      ``used_for_retrieval=False``) for transparency, but it never influences
      which chunks are fetched. A failed or uncertain detection must never block
      or misdirect a normal answer.

3. Image supplied, detector confidence **at or above** the threshold
   -> retrieve with a Chroma metadata filter, ``where={"boss": <detected class>}``,
      so the boss's own wiki chunks are preferred. If that filter yields fewer
      than ``TOP_K`` chunks -- which happens for bosses with thin wiki coverage,
      e.g. Divine Dragon has a single page -- fall back to an unfiltered
      similarity search to fill the remainder rather than returning a short or
      empty context window.

The fallback in case 3 lives in ``RetrievalService.retrieve`` (see
``app/services/retrieval.py``); this module decides *whether* to pass a boss in.
"""

from __future__ import annotations

import logging

import ollama

from app.core.config import Settings
from app.schemas.query import DetectionResult
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

# Exact sentence the model must produce when the context is insufficient. A fixed
# string is trivially checkable in evaluation, unlike an open-ended non-answer.
REFUSAL = "I don't know based on the provided sources."

SYSTEM_PROMPT = (
    "You are a Sekiro: Shadows Die Twice lore assistant. "
    "Answer using ONLY the numbered context passages supplied by the user. "
    "Do not use any outside or memorised knowledge about Sekiro, and do not "
    "speculate beyond the passages. "
    f'If the passages do not contain the answer, reply exactly: "{REFUSAL}" '
    "Cite the passages you rely on inline as [1], [2], etc."
)


class GenerationService:
    """Builds grounded prompts and calls the local Ollama model."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: ollama.Client | None = None

    def load(self) -> None:
        """Create the Ollama client once at startup.

        Deliberately does not fail when the server is down: the app should still
        start and serve ``/health`` so the problem is diagnosable, rather than
        crash-looping on boot.
        """
        self._client = ollama.Client(host=self._settings.ollama_host)
        logger.info(
            "Ollama client ready: model=%s host=%s",
            self._settings.ollama_model,
            self._settings.ollama_host,
        )

    @property
    def client(self) -> ollama.Client:
        if self._client is None:
            raise RuntimeError("GenerationService.load() has not been called")
        return self._client

    def is_model_available(self) -> tuple[bool, str]:
        """Check the configured model is actually pulled, for /health."""
        try:
            models = [m.get("model", "") for m in self.client.list().get("models", [])]
        except Exception as exc:
            return False, f"Ollama unreachable at {self._settings.ollama_host} ({exc.__class__.__name__})"

        wanted = self._settings.ollama_model.split(":")[0]
        if not any(name.split(":")[0] == wanted for name in models):
            return False, f"{self._settings.ollama_model!r} is not pulled (available: {models or 'none'})"
        return True, "ok"

    @staticmethod
    def build_prompt(question: str, chunks: list) -> str:
        """Render retrieved chunks as a numbered, citable context block."""
        blocks = []
        for i, chunk in enumerate(chunks, 1):
            boss = f", boss={chunk.boss}" if chunk.boss else ""
            blocks.append(f"[{i}] (source: {chunk.label}{boss})\n{chunk.text}")
        return "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"

    def generate(self, question: str, chunks: list) -> str:
        """One grounded generation call against the retrieved chunks."""
        response = self.client.chat(
            model=self._settings.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.build_prompt(question, chunks)},
            ],
            options={"temperature": 0.1},
        )
        return response["message"]["content"].strip()


def resolve_boss_filter(
    detection: DetectionResult | None,
    threshold: float,
) -> str | None:
    """Decide whether a detection should constrain retrieval.

    Returns the boss class name to filter on, or ``None`` for an unfiltered
    search. Cases 1 and 2 of the module docstring both return ``None``; keeping
    the rule in one function makes it directly testable.
    """
    if detection is None:
        return None
    if detection.confidence < threshold:
        return None
    return detection.boss
