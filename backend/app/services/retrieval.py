"""Vector store access.

The Chroma collection and the embedding model are loaded **once**, at application
startup (see `app.main.lifespan`), and reused for every request. Query embedding
and similarity search are the only per-request work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import chromadb

from app.core.config import Settings

logger = logging.getLogger(__name__)

# Chroma metadata cannot store `None`, so the notebook writes "" for chunks that
# do not belong to a boss page. Keep these in sync.
NO_BOSS = ""


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved passage, carrying the metadata written by the notebook."""

    text: str
    source: str
    section: str
    boss: str
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity, given the collection uses cosine space."""
        return 1.0 - self.distance

    @property
    def label(self) -> str:
        """Human-readable citation label, e.g. `Great Shinobi - Owl > Phase 1`."""
        return f"{self.source} > {self.section}" if self.section else self.source


class RetrievalService:
    """Loads the persisted store and answers nearest-neighbour queries."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embedder = None
        self._collection = None

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        """Load the embedding model and open the persisted Chroma collection.

        Called once from the FastAPI lifespan. Raises if the store is missing,
        which is deliberate: a backend that silently serves an empty index would
        look healthy while being useless.
        """
        from sentence_transformers import SentenceTransformer

        store_dir = self._settings.vector_store_dir
        if not store_dir.exists():
            raise RuntimeError(
                f"Vector store not found at {store_dir}. Run "
                "notebooks/rag_pipeline.ipynb and copy vector_store_export/ "
                "into backend/data/vector_store/ (see README)."
            )

        logger.info("Loading embedding model %s", self._settings.embedding_model)
        self._embedder = SentenceTransformer(self._settings.embedding_model, device="cpu")

        logger.info("Opening Chroma store at %s", store_dir)
        client = chromadb.PersistentClient(path=str(store_dir))
        self._collection = client.get_collection(self._settings.vector_store_collection)
        logger.info("Vector store ready: %d chunks", self._collection.count())

    @property
    def is_loaded(self) -> bool:
        return self._collection is not None and self._embedder is not None

    @property
    def chunk_count(self) -> int:
        return self._collection.count() if self._collection else 0

    @property
    def boss_classes(self) -> list[str]:
        """Distinct boss values present in the store (Extended Track)."""
        if not self._collection:
            return []
        metadatas = self._collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["boss"] for m in metadatas if m.get("boss")})

    # -- querying ----------------------------------------------------------
    def _embed(self, text: str) -> list[float]:
        # normalize_embeddings must match the notebook, which L2-normalised
        # before storing -- otherwise cosine distances are not comparable.
        return self._embedder.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        )[0].tolist()

    def _search(self, vector: list[float], limit: int, boss: str | None) -> list[RetrievedChunk]:
        kwargs = {
            "query_embeddings": [vector],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
        }
        if boss:
            kwargs["where"] = {"boss": boss}

        result = self._collection.query(**kwargs)
        return [
            RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                section=meta.get("section", ""),
                boss=meta.get("boss", NO_BOSS),
                distance=dist,
            )
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        boss: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k passages for `question`.

        When `boss` is given, results are restricted to that boss's chunks first,
        and an unfiltered similarity search backfills any shortfall. That
        fallback matters: a boss with thin wiki coverage (Divine Dragon has a
        single page) would otherwise return an empty context window and force the
        model to refuse a question it could have answered.

        `boss=None` is the Core Track path -- pure similarity search.
        """
        if not self.is_loaded:
            raise RuntimeError("RetrievalService.load() has not been called")

        limit = top_k or self._settings.top_k
        vector = self._embed(question)

        chunks: list[RetrievedChunk] = []
        seen: set[str] = set()

        if boss:
            for chunk in self._search(vector, limit, boss=boss):
                chunks.append(chunk)
                seen.add(chunk.text)
            logger.debug(
                "Boss-filtered retrieval for %r on %r: %d/%d hits",
                question, boss, len(chunks), limit,
            )

        if len(chunks) < limit:
            for chunk in self._search(vector, limit + len(seen), boss=None):
                if chunk.text in seen:
                    continue
                chunks.append(chunk)
                if len(chunks) >= limit:
                    break

        return chunks[:limit]
