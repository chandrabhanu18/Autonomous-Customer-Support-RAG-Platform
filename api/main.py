from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request

from api.schemas import (
    EvaluateResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from classification.intent_classifier import IntentClassifier
from config import settings
from db import create_connection
from evaluation.evaluator import PipelineEvaluator
from evaluation.faithfulness import FaithfulnessEvaluator
from evaluation.relevance import RelevanceEvaluator
from feedback.feedback_store import FeedbackStore
from generation.prompt_builder import PromptBuilder
from generation.response_generator import ResponseGenerator
from ingestion.embedder import Embedder
from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = create_connection()
    app.state.conn = conn
    app.state.document_loader = DocumentLoader()
    app.state.chunker = DocumentChunker(settings.chunk_size, settings.chunk_overlap)
    app.state.embedder = Embedder(settings.embedding_model)
    app.state.vector_store = VectorStore(conn)
    app.state.bm25_retriever = BM25Retriever(conn)
    app.state.hybrid_retriever = HybridRetriever(app.state.vector_store, app.state.bm25_retriever, settings.hybrid_alpha)
    app.state.intent_classifier = IntentClassifier(settings.generation_model)
    app.state.prompt_builder = PromptBuilder()
    app.state.response_generator = ResponseGenerator(settings.generation_model)
    app.state.faithfulness_evaluator = FaithfulnessEvaluator(settings.generation_model)
    app.state.relevance_evaluator = RelevanceEvaluator(settings.generation_model)
    app.state.pipeline_evaluator = PipelineEvaluator(app.state.faithfulness_evaluator, app.state.relevance_evaluator, conn)
    app.state.feedback_store = FeedbackStore(conn)
    try:
        yield
    finally:
        conn.close()


app = FastAPI(title="IntelliSupport", lifespan=lifespan)


def _ensure_app_state(app: FastAPI):
    state = app.state
    if all(
        hasattr(state, attribute)
        for attribute in (
            "conn",
            "document_loader",
            "chunker",
            "embedder",
            "vector_store",
            "bm25_retriever",
            "hybrid_retriever",
            "intent_classifier",
            "prompt_builder",
            "response_generator",
            "faithfulness_evaluator",
            "relevance_evaluator",
            "pipeline_evaluator",
            "feedback_store",
        )
    ):
        return state

    try:
        conn = create_connection()
    except Exception:
        return None

    state.conn = conn
    state.document_loader = DocumentLoader()
    state.chunker = DocumentChunker(settings.chunk_size, settings.chunk_overlap)
    state.embedder = Embedder(settings.embedding_model)
    state.vector_store = VectorStore(conn)
    state.bm25_retriever = BM25Retriever(conn)
    state.hybrid_retriever = HybridRetriever(state.vector_store, state.bm25_retriever, settings.hybrid_alpha)
    state.intent_classifier = IntentClassifier(settings.generation_model)
    state.prompt_builder = PromptBuilder()
    state.response_generator = ResponseGenerator(settings.generation_model)
    state.faithfulness_evaluator = FaithfulnessEvaluator(settings.generation_model)
    state.relevance_evaluator = RelevanceEvaluator(settings.generation_model)
    state.pipeline_evaluator = PipelineEvaluator(state.faithfulness_evaluator, state.relevance_evaluator, conn)
    state.feedback_store = FeedbackStore(conn)
    return state


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest, http_request: Request):
    state = _ensure_app_state(http_request.app)
    if state is None:
        raise HTTPException(status_code=503, detail="service unavailable")
    query_id = f"qry_{uuid4().hex[:8]}"
    response_id = f"rsp_{uuid4().hex[:8]}"
    intent_result = state.intent_classifier.classify(request.query)
    query_embedding = state.embedder.embed_text(request.query)
    top_k = max(1, request.top_k or settings.top_k)
    retrieved_chunks = state.hybrid_retriever.retrieve_with_reranking(request.query, query_embedding, top_k=top_k)
    if retrieved_chunks:
        messages = state.prompt_builder.build_rag_prompt(request.query, retrieved_chunks, intent_result)
        fallback_messages = messages
    else:
        messages = state.prompt_builder.build_clarification_prompt(request.query, intent_result)
        fallback_messages = messages
    generated = state.response_generator.generate_with_fallback(messages, fallback_messages)
    retrieved_chunk_ids = [chunk.chunk_id for chunk in retrieved_chunks]
    with state.conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO intellisupport.queries (query_id, raw_query, intent, intent_confidence)
            VALUES (%s, %s, %s, %s)
            """,
            (query_id, request.query, intent_result.intent, intent_result.confidence),
        )
        cursor.execute(
            """
            INSERT INTO intellisupport.responses (response_id, query_id, response_text, retrieved_chunk_ids)
            VALUES (%s, %s, %s, %s)
            """,
            (response_id, query_id, generated.response_text, retrieved_chunk_ids),
        )
    return QueryResponse(
        query_id=query_id,
        response_id=response_id,
        response_text=generated.response_text,
        intent=intent_result.intent,
        intent_confidence=intent_result.confidence,
        retrieved_chunk_ids=retrieved_chunk_ids,
        faithfulness_score=None,
        relevance_score=None,
    )


@app.post("/evaluate/{response_id}", response_model=EvaluateResponse)
def evaluate_endpoint(response_id: str, http_request: Request):
    state = _ensure_app_state(http_request.app)
    if state is None:
        raise HTTPException(status_code=503, detail="service unavailable")
    with state.conn.cursor() as cursor:
        cursor.execute("SELECT query_id FROM intellisupport.responses WHERE response_id = %s", (response_id,))
        row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="response not found")
    report = state.pipeline_evaluator.evaluate_response(row[0], response_id)
    return EvaluateResponse(
        response_id=response_id,
        faithfulness_score=report.faithfulness_score,
        relevance_score=report.relevance_score,
        combined_score=report.combined_score,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback_endpoint(request: FeedbackRequest, http_request: Request):
    state = _ensure_app_state(http_request.app)
    if state is None:
        raise HTTPException(status_code=503, detail="service unavailable")
    feedback_id = state.feedback_store.store_feedback(request.response_id, request.rating, request.comment)
    return FeedbackResponse(feedback_id=feedback_id, message="Feedback recorded")


@app.get("/feedback/summary/{response_id}")
def feedback_summary_endpoint(response_id: str, http_request: Request):
    state = _ensure_app_state(http_request.app)
    if state is None:
        raise HTTPException(status_code=503, detail="service unavailable")
    return state.feedback_store.get_feedback_summary(response_id)


@app.get("/health", response_model=HealthResponse)
def health_endpoint(http_request: Request):
    state = _ensure_app_state(http_request.app)
    db_connected = False
    chunks_indexed = 0
    try:
        if state is None:
            raise RuntimeError("database unavailable")
        with state.conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM intellisupport.chunks")
            chunks_indexed = int(cursor.fetchone()[0])
        db_connected = True
    except Exception:
        db_connected = False
    return HealthResponse(status="ok" if db_connected else "degraded", db_connected=db_connected, chunks_indexed=chunks_indexed)
