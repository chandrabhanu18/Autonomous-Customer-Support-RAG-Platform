from __future__ import annotations

from rank_bm25 import BM25Okapi

from retrieval.vector_store import RetrievedChunk
from utils import tokenize


class BM25Retriever:
    def __init__(self, conn):
        self.conn = conn
        self.chunks: list[dict] = []
        self.tokenized_chunks: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self.rebuild_index(conn)

    def _load_chunks(self, conn) -> None:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id, doc_id, content, chunk_index, token_count, metadata
                FROM intellisupport.chunks
                ORDER BY chunk_index ASC, chunk_id ASC
                """
            )
            rows = cursor.fetchall()
        self.chunks = [
            {
                "chunk_id": row[0],
                "doc_id": row[1],
                "content": row[2],
                "chunk_index": row[3],
                "token_count": row[4],
                "metadata": row[5] or {},
            }
            for row in rows
        ]
        self.tokenized_chunks = [tokenize(chunk["content"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks) if self.tokenized_chunks else None

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.bm25 or not self.chunks:
            return []
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        if len(scores) == 0:
            return []
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
        if not ranked:
            return []
        max_score = max(scores[index] for index in ranked)
        if max_score <= 0:
            return []
        results: list[RetrievedChunk] = []
        for index in ranked:
            raw_score = float(scores[index])
            results.append(
                RetrievedChunk(
                    chunk_id=self.chunks[index]["chunk_id"],
                    doc_id=self.chunks[index]["doc_id"],
                    content=self.chunks[index]["content"],
                    score=min(1.0, max(0.0, raw_score / max_score)),
                    retrieval_method="bm25",
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results

    def rebuild_index(self, conn) -> None:
        self.conn = conn
        self._load_chunks(conn)
