from __future__ import annotations

from generation.prompt_builder import PromptBuilder


def test_classify_billing_intent(services):
    result = services["classifier"].classify("How do I upgrade my subscription plan?")
    assert result.intent == "billing"


def test_classify_technical_intent(services):
    result = services["classifier"].classify("The app keeps crashing when I open a project")
    assert result.intent == "technical_issue"


def test_classify_confidence_range(services):
    result = services["classifier"].classify("Tell me more about Nexora")
    assert 0.0 <= result.confidence <= 1.0


def test_build_rag_prompt_structure(services):
    embedding = services["embedder"].embed_text("How do I connect Slack?")
    retrieved_chunks = services["hybrid_retriever"].retrieve("How do I connect Slack?", embedding, top_k=2)
    intent = services["classifier"].classify("How do I connect Slack?")
    messages = services["prompt_builder"].build_rag_prompt("How do I connect Slack?", retrieved_chunks, intent)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"


def test_prompt_contains_chunk_ids(services):
    embedding = services["embedder"].embed_text("How do I connect Slack?")
    retrieved_chunks = services["hybrid_retriever"].retrieve("How do I connect Slack?", embedding, top_k=2)
    intent = services["classifier"].classify("How do I connect Slack?")
    messages = services["prompt_builder"].build_rag_prompt("How do I connect Slack?", retrieved_chunks, intent)
    if retrieved_chunks:
        assert retrieved_chunks[0].chunk_id in messages[1]["content"]


def test_generate_response_fields(services):
    embedding = services["embedder"].embed_text("How do I connect Slack?")
    retrieved_chunks = services["hybrid_retriever"].retrieve("How do I connect Slack?", embedding, top_k=2)
    intent = services["classifier"].classify("How do I connect Slack?")
    messages = services["prompt_builder"].build_rag_prompt("How do I connect Slack?", retrieved_chunks, intent)
    response = services["response_generator"].generate(messages)
    assert response.response_text
    assert response.total_tokens > 0
