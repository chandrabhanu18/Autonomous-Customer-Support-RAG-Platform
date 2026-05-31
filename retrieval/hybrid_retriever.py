from __future__ import annotations

from retrieval.bm25_retriever import BM25Retriever
from retrieval.vector_store import RetrievedChunk, VectorStore
from utils import jaccard_similarity, tokenize


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_retriever: BM25Retriever, alpha: float = 0.7):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha

    def retrieve(self, query: str, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        vector_candidates = self.vector_store.similarity_search(query_embedding, top_k=top_k * 2)
        bm25_candidates = self.bm25_retriever.search(query, top_k=top_k * 2)

        merged: dict[str, RetrievedChunk] = {}
        vector_scores = {chunk.chunk_id: chunk for chunk in vector_candidates}
        bm25_scores = {chunk.chunk_id: chunk for chunk in bm25_candidates}
        all_ids = list(dict.fromkeys([*vector_scores.keys(), *bm25_scores.keys()]))

        for chunk_id in all_ids:
            vector_chunk = vector_scores.get(chunk_id)
            bm25_chunk = bm25_scores.get(chunk_id)
            if vector_chunk and bm25_chunk:
                score = (self.alpha * vector_chunk.score) + ((1 - self.alpha) * bm25_chunk.score)
                merged[chunk_id] = RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=vector_chunk.doc_id,
                    content=vector_chunk.content,
                    score=max(0.0, min(1.0, score)),
                    retrieval_method="hybrid",
                )
            elif vector_chunk:
                merged[chunk_id] = RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=vector_chunk.doc_id,
                    content=vector_chunk.content,
                    score=max(0.0, min(1.0, self.alpha * vector_chunk.score)),
                    retrieval_method="vector",
                )
            elif bm25_chunk:
                merged[chunk_id] = RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=bm25_chunk.doc_id,
                    content=bm25_chunk.content,
                    score=max(0.0, min(1.0, (1 - self.alpha) * bm25_chunk.score)),
                    retrieval_method="bm25",
                )

        results = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def retrieve_with_reranking(self, query: str, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        candidates = self.retrieve(query, query_embedding, top_k=top_k * 3)
        query_tokens = tokenize(query)
        reranked: list[RetrievedChunk] = []
        for chunk in candidates:
            chunk_tokens = tokenize(chunk.content)
            jaccard_score = jaccard_similarity(query_tokens, chunk_tokens)
            rerank_score = (0.8 * chunk.score) + (0.2 * jaccard_score)
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    score=max(0.0, min(1.0, rerank_score)),
                    retrieval_method=chunk.retrieval_method if chunk.retrieval_method else "hybrid",
                )
            )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]
