from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel

from classification.intent_classifier import IntentClassifier
from config import settings
from db import create_connection
from evaluation.faithfulness import FaithfulnessEvaluator
from evaluation.relevance import RelevanceEvaluator
from generation.prompt_builder import PromptBuilder
from generation.response_generator import ResponseGenerator
from ingestion.embedder import Embedder
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_store import RetrievedChunk, VectorStore


class EvaluationReport(BaseModel):
    query_id: str
    response_id: str
    faithfulness_score: float
    relevance_score: float
    combined_score: float


class BenchmarkReport(BaseModel):
    total_cases: int
    avg_faithfulness: float
    avg_relevance: float
    avg_combined: float
    retrieval_hit_rate: float
    intent_accuracy: float


class PipelineEvaluator:
    def __init__(self, faithfulness_evaluator: FaithfulnessEvaluator, relevance_evaluator: RelevanceEvaluator, conn):
        self.faithfulness_evaluator = faithfulness_evaluator
        self.relevance_evaluator = relevance_evaluator
        self.conn = conn

    def _load_retrieved_chunks(self, chunk_ids: list[str]) -> list[RetrievedChunk]:
        if not chunk_ids:
            return []
        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id, doc_id, content
                FROM intellisupport.chunks
                WHERE chunk_id = ANY(%s)
                """,
                (chunk_ids,),
            )
            rows = cursor.fetchall()
        row_map = {row[0]: row for row in rows}
        return [
            RetrievedChunk(chunk_id=row_map[chunk_id][0], doc_id=row_map[chunk_id][1], content=row_map[chunk_id][2], score=1.0, retrieval_method="hybrid")
            for chunk_id in chunk_ids
            if chunk_id in row_map
        ]

    def evaluate_response(self, query_id: str, response_id: str) -> EvaluationReport:
        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT q.raw_query, r.response_text, r.retrieved_chunk_ids
                FROM intellisupport.queries q
                JOIN intellisupport.responses r ON r.query_id = q.query_id
                WHERE q.query_id = %s AND r.response_id = %s
                """,
                (query_id, response_id),
            )
            row = cursor.fetchone()
        if not row:
            raise ValueError("query_id or response_id not found")
        query, response_text, retrieved_chunk_ids = row
        retrieved_chunks = self._load_retrieved_chunks(retrieved_chunk_ids or [])
        faithfulness = self.faithfulness_evaluator.evaluate(response_text, retrieved_chunks)
        relevance = self.relevance_evaluator.evaluate(query, retrieved_chunks)
        combined_score = (faithfulness.faithfulness_score + relevance.relevance_score) / 2
        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE intellisupport.responses
                SET faithfulness_score = %s, relevance_score = %s
                WHERE response_id = %s
                """,
                (faithfulness.faithfulness_score, relevance.relevance_score, response_id),
            )
        return EvaluationReport(
            query_id=query_id,
            response_id=response_id,
            faithfulness_score=faithfulness.faithfulness_score,
            relevance_score=relevance.relevance_score,
            combined_score=combined_score,
        )

    def run_benchmark(self, test_cases: list[dict]) -> BenchmarkReport:
        classifier = IntentClassifier(model=settings.generation_model)
        embedder = Embedder(model=settings.embedding_model)
        vector_store = VectorStore(self.conn)
        bm25_retriever = BM25Retriever(self.conn)
        hybrid_retriever = HybridRetriever(vector_store, bm25_retriever, alpha=settings.hybrid_alpha)
        prompt_builder = PromptBuilder()
        response_generator = ResponseGenerator(model=settings.generation_model)

        faithfulness_scores: list[float] = []
        relevance_scores: list[float] = []
        retrieval_hits = 0
        intent_matches = 0

        for case in test_cases:
            query = case["query"]
            expected_doc_ids = case["expected_doc_ids"]
            expected_intent = case["expected_intent"]

            intent_result = classifier.classify(query)
            if intent_result.intent == expected_intent:
                intent_matches += 1

            query_embedding = embedder.embed_text(query)
            retrieved_chunks = hybrid_retriever.retrieve_with_reranking(query, query_embedding, top_k=settings.top_k)
            if any(chunk.doc_id in expected_doc_ids for chunk in retrieved_chunks):
                retrieval_hits += 1

            messages = (
                prompt_builder.build_clarification_prompt(query, intent_result)
                if not retrieved_chunks
                else prompt_builder.build_rag_prompt(query, retrieved_chunks, intent_result)
            )
            generated = response_generator.generate(messages)
            faithfulness = self.faithfulness_evaluator.evaluate(generated.response_text, retrieved_chunks)
            relevance = self.relevance_evaluator.evaluate(query, retrieved_chunks)
            faithfulness_scores.append(faithfulness.faithfulness_score)
            relevance_scores.append(relevance.relevance_score)

        total_cases = len(test_cases) or 1
        avg_faithfulness = sum(faithfulness_scores) / total_cases
        avg_relevance = sum(relevance_scores) / total_cases
        avg_combined = (avg_faithfulness + avg_relevance) / 2
        return BenchmarkReport(
            total_cases=len(test_cases),
            avg_faithfulness=avg_faithfulness,
            avg_relevance=avg_relevance,
            avg_combined=avg_combined,
            retrieval_hit_rate=retrieval_hits / total_cases,
            intent_accuracy=intent_matches / total_cases,
        )
