from __future__ import annotations

from pathlib import Path

import pytest

from config import settings
from db import create_connection
from evaluation.evaluator import PipelineEvaluator
from evaluation.faithfulness import FaithfulnessEvaluator
from evaluation.relevance import RelevanceEvaluator
from feedback.feedback_store import FeedbackStore
from generation.prompt_builder import PromptBuilder
from generation.response_generator import ResponseGenerator
from ingestion.chunker import DocumentChunker
from ingestion.embedder import Embedder
from ingestion.loader import DocumentLoader
from ingestion.seed_data import SEED_DOCUMENTS
from classification.intent_classifier import IntentClassifier
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_store import VectorStore


@pytest.fixture(scope="session")
def prepared_conn():
    try:
        conn = create_connection()
    except Exception as exc:
        pytest.skip(f"Database unavailable: {exc}")

    migration_path = Path(__file__).resolve().parents[1] / "database" / "migrations" / "001_initial.sql"
    migration_sql = migration_path.read_text(encoding="utf-8")
    with conn.cursor() as cursor:
        cursor.execute(migration_sql)
        cursor.execute(
            """
            TRUNCATE TABLE
                intellisupport.feedback,
                intellisupport.responses,
                intellisupport.queries,
                intellisupport.chunks,
                intellisupport.documents
            RESTART IDENTITY CASCADE
            """
        )

    loader = DocumentLoader()
    chunker = DocumentChunker(settings.chunk_size, settings.chunk_overlap)
    embedder = Embedder(settings.embedding_model)
    documents = loader.load_batch(SEED_DOCUMENTS)
    loader.save_to_db(documents, conn)
    chunks = chunker.chunk_batch(documents)
    embedder.embed_and_store_chunks(chunks, conn)

    yield conn
    conn.close()


@pytest.fixture(scope="session")
def services(prepared_conn):
    vector_store = VectorStore(prepared_conn)
    bm25_retriever = BM25Retriever(prepared_conn)
    hybrid_retriever = HybridRetriever(vector_store, bm25_retriever, settings.hybrid_alpha)
    embedder = Embedder(settings.embedding_model)
    classifier = IntentClassifier(settings.generation_model)
    prompt_builder = PromptBuilder()
    response_generator = ResponseGenerator(settings.generation_model)
    faithfulness_evaluator = FaithfulnessEvaluator(settings.generation_model)
    relevance_evaluator = RelevanceEvaluator(settings.generation_model)
    pipeline_evaluator = PipelineEvaluator(faithfulness_evaluator, relevance_evaluator, prepared_conn)
    feedback_store = FeedbackStore(prepared_conn)
    return {
        "conn": prepared_conn,
        "vector_store": vector_store,
        "bm25_retriever": bm25_retriever,
        "hybrid_retriever": hybrid_retriever,
        "embedder": embedder,
        "classifier": classifier,
        "prompt_builder": prompt_builder,
        "response_generator": response_generator,
        "faithfulness_evaluator": faithfulness_evaluator,
        "relevance_evaluator": relevance_evaluator,
        "pipeline_evaluator": pipeline_evaluator,
        "feedback_store": feedback_store,
    }
