from __future__ import annotations

from pydantic import BaseModel

from db import vector_to_pgvector_literal


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    score: float
    retrieval_method: str


class VectorStore:
    def __init__(self, conn):
        self.conn = conn

    def _query(self, query_embedding: list[float], top_k: int):
        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id, doc_id, content, 1 - (embedding <=> %s::vector) AS similarity
                FROM intellisupport.chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vector_to_pgvector_literal(query_embedding), vector_to_pgvector_literal(query_embedding), top_k),
            )
            return cursor.fetchall()

    def similarity_search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        rows = self._query(query_embedding, top_k)
        results = [
            RetrievedChunk(
                chunk_id=row[0],
                doc_id=row[1],
                content=row[2],
                score=max(0.0, min(1.0, float(row[3]))),
                retrieval_method="vector",
            )
            for row in rows
        ]
        results.sort(key=lambda item: item.score, reverse=True)
        return results

    def similarity_search_with_threshold(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.75,
    ) -> list[RetrievedChunk]:
        results = self.similarity_search(query_embedding, top_k=top_k)
        return [result for result in results if result.score >= threshold]
