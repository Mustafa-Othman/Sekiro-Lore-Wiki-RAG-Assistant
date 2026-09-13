# Backend image for the Sekiro wiki RAG assistant.
#
# Build from the repository root so the vector store is in context:
#
#     docker build -f backend/Dockerfile -t sekiro-rag-backend .
#
# Run (see the note on OLLAMA_HOST below):
#
#     docker run --rm -p 8000:8000 --add-host=host.docker.internal:host-gateway \
#       -e OLLAMA_HOST=http://host.docker.internal:11434 sekiro-rag-backend

# Python 3.12 rather than the 3.14 used for local development: torch and
# sentence-transformers publish manylinux wheels for 3.12, which keeps the build
# from needing a compiler toolchain.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# Dependencies first, so editing application code does not invalidate this layer.
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image. The model must match the one used in
# notebooks/rag_pipeline.ipynb, and downloading it here means the container needs
# no network access at request time (or at all, once built).
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('embedding model cached')"

COPY backend/app ./app
COPY backend/tests ./tests

# The persisted Chroma store, produced by notebooks/rag_pipeline.ipynb (5.7).
# Baking it in keeps the image self-contained; mount a volume instead if you
# want to swap indexes without rebuilding.
COPY backend/data/vector_store ./data/vector_store

# Run as a non-root user. The store is only ever read, so ownership is enough.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Note for anyone running this: with Ollama on the *host*, OLLAMA_HOST must be
# http://host.docker.internal:11434 -- inside the container "localhost" is the
# container itself, where nothing is listening on 11434.
ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    VECTOR_STORE_PATH=./data/vector_store \
    VECTOR_STORE_COLLECTION=sekiro_wiki \
    FRONTEND_ORIGIN=http://localhost:8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
