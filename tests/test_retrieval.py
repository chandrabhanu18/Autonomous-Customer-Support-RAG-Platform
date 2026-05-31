from __future__ import annotations


def test_vector_similarity_search_returns_k_results(services):
    embedding = services["embedder"].embed_text("How do I pay for Nexora?")
    results = services["vector_store"].similarity_search(embedding, top_k=5)
    assert len(results) == 5


def test_vector_similarity_scores_range(services):
    embedding = services["embedder"].embed_text("How do I pay for Nexora?")
    results = services["vector_store"].similarity_search(embedding, top_k=5)
    assert all(0.0 <= result.score <= 1.0 for result in results)


def test_bm25_search_returns_results(services):
    results = services["bm25_retriever"].search("billing subscription", top_k=5)
    assert len(results) >= 1


def test_bm25_keyword_relevance(services):
    results = services["bm25_retriever"].search("two factor authentication", top_k=5)
    assert any(result.doc_id == "doc_008" for result in results[:3])


def test_hybrid_retriever_score_range(services):
    embedding = services["embedder"].embed_text("How do I connect Slack?")
    results = services["hybrid_retriever"].retrieve("How do I connect Slack?", embedding, top_k=5)
    assert all(0.0 <= result.score <= 1.0 for result in results)


def test_hybrid_deduplication(services):
    embedding = services["embedder"].embed_text("How do I connect Slack?")
    results = services["hybrid_retriever"].retrieve("How do I connect Slack?", embedding, top_k=10)
    chunk_ids = [result.chunk_id for result in results]
    assert len(chunk_ids) == len(set(chunk_ids))
