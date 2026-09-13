"""Application settings, loaded from environment / `.env` once at import."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view of the backend's environment variables.

    Values come from the process environment, falling back to a `.env` file in
    the backend directory. See `.env.example` for the documented set.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Local LLM ---------------------------------------------------------
    # The exact Ollama tag, including any registry prefix and the ":latest" suffix
    # omitted. Must match a pulled model or /health reports the LLM as unavailable.
    ollama_model: str = "qcwind/qwen2.5-7B-instruct-Q4_K_M"
    ollama_host: str = "http://localhost:11434"

    # --- Vector store ------------------------------------------------------
    vector_store_path: str = "./data/vector_store"
    vector_store_collection: str = "sekiro_wiki"
    # Must match the model used in notebooks/rag_pipeline.ipynb, or query
    # embeddings will be incompatible with the stored chunk embeddings.
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 4

    # --- CORS --------------------------------------------------------------
    frontend_origin: str = "http://localhost:8501"

    # --- Extended Track: YOLO boss detection -------------------------------
    # When unset (or the file is missing) the app runs the Core Track pipeline
    # only -- detection is skipped, never fatal.
    yolo_model_path: str | None = None
    yolo_confidence_threshold: float = 0.5

    @property
    def vector_store_dir(self) -> Path:
        return Path(self.vector_store_path).resolve()

    @property
    def yolo_model_file(self) -> Path | None:
        if not self.yolo_model_path:
            return None
        return Path(self.yolo_model_path).resolve()


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the environment is read once per process."""
    return Settings()
