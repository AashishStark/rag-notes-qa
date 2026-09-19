"""
FastAPI REST API for Handwritten Notes RAG Pipeline
--------------------------------------------------
Exposes the RAGPipeline via standard REST endpoints with structured logging
and in-memory observability metrics.

Endpoints:
  - POST /query   : Submits a question and returns grounded answer with sources
  - GET  /health  : Health check confirming vector database state and chunk counts
  - GET  /metrics : Real-time in-memory performance and reliability metrics
  - GET  /docs    : Interactive Swagger API documentation

Usage:
  Direct Python run : python api.py
  Development mode  : uvicorn api:app --reload
"""

import sys
import time
import re
import logging
from threading import Lock
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
import uvicorn

from rag_pipeline import RAGPipeline


# =============================================================================
# ARCHITECTURAL NOTE: Why Logging to stdout Matters for Containers & K8s
# =============================================================================
# In containerized environments (Docker & Kubernetes), apps must treat logs as
# continuous event streams routed directly to stdout/stderr rather than managing
# local log files.
#
# 1. Twelve-Factor App (Factor XI - Logs):
#    A containerized app should never concern itself with routing or storage of
#    its output stream. It should write unbuffered logs to stdout.
#
# 2. Docker Daemon Log Driver:
#    Docker automatically captures container stdout/stderr streams and forwards
#    them to the configured logging driver (json-file, syslog, fluentd, awslogs).
#    This makes `docker logs <container>` work out of the box.
#
# 3. Kubernetes & Cloud-Native Observability:
#    Kubelet redirects container stdout/stderr to `/var/log/pods/` on the node.
#    Node-level DaemonSets (e.g., Fluent Bit, Promtail, Datadog Agent) collect,
#    parse, and ship these streams to centralized platforms (Elasticsearch, Loki,
#    CloudWatch) without mounting shared volumes into app containers.
#
# 4. Ephemeral Storage Protection:
#    Writing log files inside a container writes to the union filesystem (overlay2),
#    causing container disk bloat, risk of DiskPressure evictions in K8s, and
#    loss of logs when pods crash or restart.
# =============================================================================

# Configure structured root logger outputting strictly to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("rag_api")

# Global reference to pipeline initialized once at startup
pipeline: Optional[RAGPipeline] = None


# =============================================================================
# In-Memory Metrics Tracker
# =============================================================================

class MetricsTracker:
    """
    Thread-safe in-memory metrics tracker for monitoring query traffic,
    error counts, and processing latency.
    """
    def __init__(self):
        self._lock = Lock()
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0

    def record_request(self, latency_ms: float, success: bool = True):
        with self._lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
                self.total_latency_ms += latency_ms
            else:
                self.failed_requests += 1

    def get_metrics(self) -> dict:
        with self._lock:
            uptime = round(time.time() - self.start_time, 2)
            avg_latency = (
                round(self.total_latency_ms / self.successful_requests, 2)
                if self.successful_requests > 0
                else 0.0
            )
            return {
                "uptime_seconds": uptime,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "average_latency_ms": avg_latency,
                "total_latency_ms": round(self.total_latency_ms, 2),
            }


metrics = MetricsTracker()


def sanitize_secrets(message: str) -> str:
    """
    Sanitizes log messages by masking potential Gemini / Google API keys
    to prevent accidental credential leakage in logs.
    """
    # Masks Google AI Studio / Gemini API key patterns (AIza... or AQ....)
    cleaned = re.sub(r"AIza[0-9A-Za-z\-_]{35}", "[REDACTED_API_KEY]", message)
    cleaned = re.sub(r"AQ\.[0-9A-Za-z\-_]{30,}", "[REDACTED_API_KEY]", cleaned)
    return cleaned


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Manager:
    Initializes the RAGPipeline once at startup so the embedding model
    and ChromaDB connection are shared across all requests.
    """
    global pipeline
    logger.info("Initializing RAG pipeline at startup...")
    try:
        pipeline = RAGPipeline()
        chunk_count = pipeline.collection.count() if pipeline.collection else 0
        logger.info(f"RAG pipeline successfully initialized with {chunk_count} indexed chunks.")
    except Exception as exc:
        safe_msg = sanitize_secrets(str(exc))
        logger.error(f"Failed to initialize RAG pipeline: {safe_msg}", exc_info=True)
        raise exc

    yield  # Application serves requests

    logger.info("Shutting down RAG API server...")


# Initialize FastAPI application
app = FastAPI(
    title="Handwritten Study Notes RAG API",
    description="Retrieval-Augmented Generation API with structured logging and metrics.",
    version="1.1.0",
    lifespan=lifespan,
)


# =============================================================================
# Pydantic Request & Response Models
# =============================================================================

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The question to ask based on handwritten study notes.",
        examples=["What is AWS EBS and what are its use cases?"]
    )
    k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of most relevant chunks to retrieve from ChromaDB.",
        examples=[4]
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or contain only whitespace.")
        return cleaned


class SourceChunk(BaseModel):
    chunk_id: str
    source_file: str
    chunk_index: int
    topic_heading: str
    distance: Optional[float] = None
    text: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    database: str
    collection_name: str
    total_chunks_indexed: int
    embedding_model: str
    gemini_model: str


class MetricsResponse(BaseModel):
    uptime_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency_ms: float
    total_latency_ms: float


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint pointing to interactive documentation and monitoring."""
    return {
        "message": "Handwritten Study Notes RAG API is online.",
        "documentation": "/docs",
        "health_check": "/health",
        "metrics_endpoint": "/metrics",
        "query_endpoint": "/query",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["System"],
)
async def health_check():
    """
    Container liveness and readiness probe.
    Confirms ChromaDB is loaded and returns chunk metrics.
    """
    if pipeline is None or pipeline.collection is None:
        logger.warning("Health check failed: RAG pipeline or ChromaDB not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline or vector database is not initialized.",
        )

    try:
        chunk_count = pipeline.collection.count()
        return HealthResponse(
            status="healthy",
            database="ChromaDB (Persistent)",
            collection_name=pipeline.collection_name,
            total_chunks_indexed=chunk_count,
            embedding_model="all-MiniLM-L6-v2 (Local)",
            gemini_model=pipeline.gemini_model,
        )
    except Exception as exc:
        safe_msg = sanitize_secrets(str(exc))
        logger.error(f"Health probe error: {safe_msg}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health probe error: {safe_msg}",
        )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    tags=["Observability"],
)
async def get_metrics():
    """
    Returns basic service observability metrics:
    - Uptime in seconds
    - Total requests handled
    - Successful vs failed requests
    - Average processing latency in milliseconds
    """
    return metrics.get_metrics()


@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["RAG"],
)
async def query_rag(req: QueryRequest):
    """
    Processes a query through the RAG pipeline with latency monitoring & logging:
    1. Logs incoming question and retrieval parameters.
    2. Measures semantic search and Gemini generation latency.
    3. Records request metrics and logs completion or error details.
    """
    start_time = time.perf_counter()
    logger.info(f"Incoming /query: question='{req.question}', k={req.k}")

    if pipeline is None:
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics.record_request(latency_ms=latency_ms, success=False)
        logger.error("Query rejected: RAG pipeline is not initialized yet.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialized yet.",
        )

    try:
        # Execute retrieval and generation
        result = pipeline.generate_answer(query=req.question, k=req.k)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Check for upstream error string returned by pipeline
        answer_text = result.get("answer", "")
        if answer_text.startswith("[Error"):
            safe_err = sanitize_secrets(answer_text)
            metrics.record_request(latency_ms=latency_ms, success=False)
            logger.error(f"Upstream generation failed: {safe_err} (latency={latency_ms:.2f}ms)")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=safe_err,
            )

        # Format sources
        formatted_sources = [
            SourceChunk(
                chunk_id=s.get("chunk_id", ""),
                source_file=s.get("source_file", ""),
                chunk_index=s.get("chunk_index", 0),
                topic_heading=s.get("topic_heading", "General"),
                distance=round(float(s["distance"]), 4) if "distance" in s and s["distance"] is not None else None,
                text=s.get("text"),
            )
            for s in result.get("sources", [])
        ]

        # Record success metrics and log outcome
        metrics.record_request(latency_ms=latency_ms, success=True)
        logger.info(
            f"Query completed: question='{req.question}', "
            f"sources_retrieved={len(formatted_sources)}, "
            f"latency={latency_ms:.2f}ms"
        )

        return QueryResponse(
            query=req.question,
            answer=answer_text,
            sources=formatted_sources,
        )

    except HTTPException:
        raise
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics.record_request(latency_ms=latency_ms, success=False)
        safe_msg = sanitize_secrets(str(exc))
        logger.error(f"Query failed ({type(exc).__name__}): {safe_msg} (latency={latency_ms:.2f}ms)")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating answer: {safe_msg}",
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting Uvicorn server on http://localhost:8000")
    logger.info("API Documentation available at: http://localhost:8000/docs")
    logger.info("Metrics endpoint available at: http://localhost:8000/metrics")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
