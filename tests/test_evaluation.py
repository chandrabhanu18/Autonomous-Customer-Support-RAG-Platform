from __future__ import annotations

BENCHMARK_TEST_CASES = [
    {
        "query": "How do I add a new member to my team?",
        "expected_doc_ids": ["doc_002"],
        "expected_intent": "account_management",
    },
    {
        "query": "What happens if I cancel my subscription?",
        "expected_doc_ids": ["doc_003"],
        "expected_intent": "billing",
    },
    {
        "query": "How do I connect Nexora to Slack?",
        "expected_doc_ids": ["doc_004"],
        "expected_intent": "integration",
    },
    {
        "query": "I forgot my password and can't log in",
        "expected_doc_ids": ["doc_008", "doc_002"],
        "expected_intent": "account_management",
    },
    {
        "query": "How do I export my project data as CSV?",
        "expected_doc_ids": ["doc_007"],
        "expected_intent": "data_and_export",
    },
    {
        "query": "The webhook I set up isn't receiving any events",
        "expected_doc_ids": ["doc_009"],
        "expected_intent": "technical_issue",
    },
    {
        "query": "Can I use custom templates for new projects?",
        "expected_doc_ids": ["doc_005"],
        "expected_intent": "feature_request",
    },
    {
        "query": "How do I enable two-factor authentication for my account?",
        "expected_doc_ids": ["doc_008"],
        "expected_intent": "account_management",
    },
]


def test_faithfulness_score_range(services):
    query = "How do I connect Nexora to Slack?"
    embedding = services["embedder"].embed_text(query)
    retrieved_chunks = services["hybrid_retriever"].retrieve(query, embedding, top_k=3)
    intent = services["classifier"].classify(query)
    messages = services["prompt_builder"].build_rag_prompt(query, retrieved_chunks, intent)
    response = services["response_generator"].generate(messages)
    result = services["faithfulness_evaluator"].evaluate(response.response_text, retrieved_chunks)
    assert 0.0 <= result.faithfulness_score <= 1.0


def test_relevance_score_range(services):
    query = "How do I connect Nexora to Slack?"
    embedding = services["embedder"].embed_text(query)
    retrieved_chunks = services["hybrid_retriever"].retrieve(query, embedding, top_k=3)
    result = services["relevance_evaluator"].evaluate(query, retrieved_chunks)
    assert 0.0 <= result.relevance_score <= 1.0


def test_benchmark_hit_rate(services):
    report = services["pipeline_evaluator"].run_benchmark(BENCHMARK_TEST_CASES)
    assert report.retrieval_hit_rate >= 0.6


def test_benchmark_intent_accuracy(services):
    report = services["pipeline_evaluator"].run_benchmark(BENCHMARK_TEST_CASES)
    assert report.intent_accuracy >= 0.75


def test_benchmark_avg_faithfulness(services):
    report = services["pipeline_evaluator"].run_benchmark(BENCHMARK_TEST_CASES)
    assert report.avg_faithfulness >= 0.6
