"""Boss detection from an uploaded screenshot (Extended Track, Section 5.5.8).

The fine-tuned YOLOv8 weights are loaded **once** at startup, exactly like the
vector store, and reused per request. Nothing is retrained at request time.

This service is optional: when Ultralytics is not installed or the weights file is
absent, `load()` logs that detection is disabled and the app continues to serve the
Core Track pipeline. Detection is an enhancement, and a missing model must not take
the RAG endpoint down with it.
"""

from __future__ import annotations

import io
import logging

from app.core.config import Settings

logger = logging.getLogger(__name__)


class DetectionService:
    """Wraps a fine-tuned YOLO model and exposes a single `detect()` call."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._weights = settings.yolo_model_file

    def load(self) -> None:
        """Load the fine-tuned weights once at startup.

        Never raises: any problem downgrades the service to disabled, and
        `is_enabled` reports that honestly through `/health`.
        """
        if self._weights is None:
            logger.info("YOLO_MODEL_PATH not set -- boss detection disabled")
            return
        if not self._weights.exists():
            logger.warning(
                "YOLO weights not found at %s -- boss detection disabled", self._weights
            )
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning(
                "ultralytics is not installed -- boss detection disabled "
                "(pip install ultralytics to enable it)"
            )
            return

        self._model = YOLO(str(self._weights))
        logger.info(
            "YOLO model loaded from %s (classes: %s)",
            self._weights,
            list(self._model.names.values()),
        )

    @property
    def is_enabled(self) -> bool:
        return self._model is not None

    def detect(self, image_bytes: bytes) -> tuple[str, float] | None:
        """Run inference on one image.

        Returns the highest-confidence `(class_name, confidence)` pair, or `None`
        when detection is disabled or the image contains no detection above
        `YOLO_CONFIDENCE_THRESHOLD`.

        The returned class name matches the `boss` metadata value written by the
        notebook, which is what makes the retrieval filter in
        `app/services/generation.py` work.
        """
        if not self.is_enabled:
            return None

        import numpy as np
        from PIL import Image

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            logger.warning("Uploaded file could not be decoded as an image")
            return None

        results = self._model.predict(np.array(image), verbose=False)
        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        confidences = boxes.conf.cpu().numpy()
        best = int(confidences.argmax())
        confidence = float(confidences[best])
        if confidence < self._settings.yolo_confidence_threshold:
            return None

        class_name = self._model.names[int(boxes.cls[best])]
        return class_name, confidence
