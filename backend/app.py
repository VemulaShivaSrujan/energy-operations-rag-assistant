"""
Energy Operations Document RAG Assistant - FastAPI Application
=============================================================
Serving Greenko renewable operations document intelligence, SOP search,
grounded generation, citation tracking, and hallucination verification.
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.rag_engine import EnergyOpsRAGPipeline, RAGResponse
from backend.evaluator import RAGEvaluator

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"

app = FastAPI(
    title="Greenko Energy Operations Document RAG Assistant",
    description="Mission-critical RAG assistant for renewable energy operations, safety, and maintenance SOPs with citations and hallucination checks.",
    version="1.0.0"
)

# Instantiate singleton pipeline
pipeline = EnergyOpsRAGPipeline(data_dir=DATA_DIR)
evaluator = RAGEvaluator(pipeline=pipeline)


# ============================================================================
# API Models
# ============================================================================

class Query(BaseModel):
    text: str = Field(..., description="Operational query from technician or control room engineer")
    top_k: Optional[int] = Field(default=3, ge=1, le=10, description="Number of context chunks to retrieve")


class FeedbackModel(BaseModel):
    query: str
    feedback: str  # 'positive' | 'negative'
    comment: Optional[str] = None


# ============================================================================
# Core API Endpoints
# ============================================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "system": "Energy Operations Document RAG Assistant",
        "total_documents": len(pipeline.documents),
        "total_chunks": len(pipeline.chunks),
        "provider": pipeline.generator.openai_key and "OpenAI" or "LocalGroundedSynthesizer"
    }


@app.get("/api/demo")
def demo():
    """Preserves starter contract with live stats."""
    return {
        "case_study": "Energy Operations Document RAG Assistant",
        "status": "production-ready RAG pipeline active",
        "domain": "Greenko Renewable Asset Operations (Wind, Solar, Hydro, Grid, BESS)",
        "expected_outcome": "A working RAG-based assistant that answers operational questions from provided documents with citations, retrieval evidence, and hallucination checks.",
        "documents_indexed": len(pipeline.documents),
        "chunks_indexed": len(pipeline.chunks),
        "sample_queries": [
            "What should be checked for inverter derating?",
            "How should gearbox vibration be investigated?",
            "What should be reviewed for generation deviation before escalation?",
            "What safety steps are required before spillway gate inspection?",
            "What actions are required if BESS cell temperature exceeds 55C?"
        ]
    }


@app.post("/api/run")
def run_case(query: Query):
    """Preserves starter contract while executing end-to-end RAG."""
    if not query.text or not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    rag_resp = pipeline.answer_query(query.text, top_k=query.top_k or 3)
    return {
        "input": query.text,
        "result": rag_resp.answer,
        "citations": rag_resp.citations,
        "verification_status": rag_resp.verification_status,
        "groundedness_score": rag_resp.groundedness_score,
        "hallucination_detected": rag_resp.hallucination_detected,
        "confidence": rag_resp.confidence_level,
        "retrieved_chunks_count": len(rag_resp.retrieved_chunks),
        "latency_ms": rag_resp.latency_breakdown_ms["total_ms"]
    }


@app.post("/api/query", response_model=RAGResponse)
def query_rag(query: Query):
    """Full enterprise endpoint returning complete retrieval evidence and hallucination check breakdown."""
    if not query.text or not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    return pipeline.answer_query(query.text, top_k=query.top_k or 3)


@app.get("/api/documents")
def list_documents():
    """Returns all indexed SOP documents and chunks for knowledge transparency."""
    docs_summary = []
    for doc in pipeline.documents:
        doc_chunks = [c for c in pipeline.chunks if c.doc_name == doc["doc_name"]]
        docs_summary.append({
            "doc_name": doc["doc_name"],
            "title": doc["title"],
            "word_count": len(doc["content"].split()),
            "chunks_count": len(doc_chunks),
            "preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
            "full_content": doc["content"]
        })
    return {
        "total_documents": len(docs_summary),
        "total_chunks": len(pipeline.chunks),
        "documents": docs_summary
    }


@app.post("/api/evaluate")
def run_evaluation():
    """Runs the quantitative benchmark against data/questions.csv."""
    return evaluator.run_benchmark()


@app.get("/api/stats")
def get_stats():
    """Returns telemetry and pipeline configuration stats."""
    return {
        "documents_count": len(pipeline.documents),
        "chunks_count": len(pipeline.chunks),
        "vocabulary_size": len(pipeline.retriever.vectorizer.vocabulary_) if pipeline.retriever.vectorizer else 0,
        "semantic_alpha": pipeline.retriever.alpha,
        "active_generator": "OpenAI (Cloud)" if pipeline.generator.openai_key else "Local Grounded Synthesizer (Zero-dependency)",
        "grounding_threshold": pipeline.verifier.grounding_threshold
    }


# ============================================================================
# Static Files & Frontend Routing
# ============================================================================

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"status": "Frontend index.html not found"}
