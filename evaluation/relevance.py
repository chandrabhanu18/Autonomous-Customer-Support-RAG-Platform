from __future__ import annotations

import json
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from config import settings
from retrieval.vector_store import RetrievedChunk
from utils import tokenize


class ChunkRelevanceScore(BaseModel):
    chunk_id: str
    score: int
    reason: str


class RelevanceResult(BaseModel):
    relevance_score: float
    chunk_scores: list[ChunkRelevanceScore]
    query: str


class RelevanceEvaluator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client: Optional[OpenAI] = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def _fallback(self, query: str, retrieved_chunks: list[RetrievedChunk]) -> RelevanceResult:
        query_tokens = set(tokenize(query))
        chunk_scores: list[ChunkRelevanceScore] = []
        total = 0
        for chunk in retrieved_chunks:
            chunk_tokens = set(tokenize(chunk.content))
            if not query_tokens or not chunk_tokens:
                score = 0
            else:
                overlap = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
                if overlap >= 0.35:
                    score = 2
                elif overlap >= 0.15:
                    score = 1
                else:
                    score = 0
            total += score
            chunk_scores.append(
                ChunkRelevanceScore(
                    chunk_id=chunk.chunk_id,
                    score=score,
                    reason=f"Lexical overlap score={score}",
                )
            )
        relevance_score = (total / (2 * len(retrieved_chunks))) if retrieved_chunks else 0.0
        return RelevanceResult(relevance_score=relevance_score, chunk_scores=chunk_scores, query=query)

    def evaluate(self, query: str, retrieved_chunks: list[RetrievedChunk]) -> RelevanceResult:
        system_prompt = (
            "You are a strict retrieval judge. Rate each chunk for relevance to the query on a 0-2 scale. "
            "0 = Not relevant, 1 = Partially relevant, 2 = Highly relevant. "
            "Return only JSON with key chunk_scores containing objects with chunk_id, score, and reason."
        )
        user_prompt = {
            "query": query,
            "chunks": [
                {"chunk_id": chunk.chunk_id, "content": chunk.content, "doc_id": chunk.doc_id}
                for chunk in retrieved_chunks
            ],
        }
        if self._client and retrieved_chunks:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(user_prompt)},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                payload = json.loads(response.choices[0].message.content or "{}")
                scores = []
                total = 0
                for item in payload.get("chunk_scores", []):
                    score = int(item.get("score", 0))
                    score = max(0, min(2, score))
                    total += score
                    scores.append(
                        ChunkRelevanceScore(
                            chunk_id=item.get("chunk_id", ""),
                            score=score,
                            reason=str(item.get("reason", "")),
                        )
                    )
                relevance_score = (total / (2 * len(scores))) if scores else 0.0
                return RelevanceResult(relevance_score=max(0.0, min(1.0, relevance_score)), chunk_scores=scores, query=query)
            except Exception:
                return self._fallback(query, retrieved_chunks)
        return self._fallback(query, retrieved_chunks)

    def evaluate_batch(self, queries_and_chunks: list[tuple[str, list[RetrievedChunk]]]) -> list[RelevanceResult]:
        return [self.evaluate(query, chunks) for query, chunks in queries_and_chunks]
