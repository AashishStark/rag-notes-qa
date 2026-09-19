# =============================================================================
# Handwritten Notes RAG API — Production Dockerfile
# =============================================================================

# 1. Official lightweight Python 3.11 slim base image
FROM python:3.11-slim

# 2. Configure Python environment flags
# - PYTHONUNBUFFERED=1 ensures real-time log output without buffering
# - PYTHONDONTWRITEBYTECODE=1 prevents creating .pyc files inside the container
# - PIP_NO_CACHE_DIR=1 minimizes image layer size
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# 3. Install lightweight system dependencies (curl for container health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Create a dedicated non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# 5. Set working directory
WORKDIR /app

# 6. Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# 7. Install Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 8. Create directory for ChromaDB vector storage and set permissions
RUN mkdir -p /app/data/chroma_db && \
    chown -R appuser:appuser /app

# 9. Copy application source code and initial corpus chunks
# Note: ChromaDB persistent vector data is excluded via .dockerignore
# and mounted at runtime via Docker volumes.
COPY api.py rag_pipeline.py ./
COPY data/chunks.json data/eval_set.json ./data/

# Ensure all copied files belong to the non-root user
RUN chown -R appuser:appuser /app

# 10. Switch to non-root user (Principle of Least Privilege)
USER appuser

# 11. Expose application port
EXPOSE 8000

# 12. Health check instruction for container orchestrators
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 13. Production entrypoint running Uvicorn
# Single worker is recommended for in-memory embedding models & ChromaDB SQLite locks
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
