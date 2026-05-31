import importlib
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from utils import deterministic_embedding


class FakeResponseItem:
    def __init__(self, embedding=None, content=None):
        self.embedding = embedding
        self.message = type("M", (), {"content": content})


class FakeEmbeddings:
    def __init__(self, model=None):
        self.model = model

    def create(self, model, input):
        data = [FakeResponseItem(embedding=deterministic_embedding(text)) for text in input]
        return type("R", (), {"data": data})


class FakeCompletions:
    def __init__(self):
        pass

    def create(self, **kwargs):
        # Provide deterministic content depending on prompt type
        messages = kwargs.get("messages") or []
        user = messages[-1]["content"] if messages else ""
        if "Return only valid JSON with keys total_claims" in messages[0]["content"] if messages else False:
            # faithfulness/relevance style judge
            content = json.dumps({"total_claims": 1, "supported_claims": 1, "unsupported_claims": 0, "reasoning": "auto"})
        elif "You are an intent classifier" in messages[0]["content"] if messages else False:
            content = json.dumps({"intent": "account_management", "confidence": 1.0})
        else:
            # generator
            content = "Based on the Nexora documentation, This is a deterministic golden response. Relevant chunks: chunk_doc_001_0."
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": content})})]})


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeOpenAI:
    def __init__(self, api_key=None):
        self.embeddings = FakeEmbeddings()
        self.chat = FakeChat()


@pytest.fixture(autouse=True)
def patch_openai():
    # Patch OpenAI class in modules that import it
    target_modules = [
        "ingestion.embedder",
        "generation.response_generator",
        "classification.intent_classifier",
        "evaluation.faithfulness",
        "evaluation.relevance",
    ]
    patches = []
    for mod in target_modules:
        m = importlib.import_module(mod)
        if hasattr(m, "OpenAI"):
            patches.append(patch.object(m, "OpenAI", FakeOpenAI))
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


def test_golden_end_to_end(tmp_path):
    # Patch the retriever to return deterministic retrieved chunks (avoid needing Postgres)
    from retrieval.vector_store import RetrievedChunk
    import retrieval.hybrid_retriever as hr

    def fake_retrieve_with_reranking(self, query, query_embedding, top_k=5):
        chunks = [
            RetrievedChunk(chunk_id="chunk_doc_001_0", doc_id="doc_001", content="Two-factor authentication is configured in account security settings.", score=1.0, retrieval_method="hybrid"),
            RetrievedChunk(chunk_id="chunk_doc_002_0", doc_id="doc_002", content="You can export projects as CSV from workspace admin.", score=0.8, retrieval_method="hybrid"),
        ]
        return chunks[:top_k]

    original = hr.HybridRetriever.retrieve_with_reranking
    hr.HybridRetriever.retrieve_with_reranking = fake_retrieve_with_reranking

    # Create a fake in-memory DB-like connection and state to avoid Postgres
    class FakeCursor:
        def __init__(self, store):
            self.store = store
            self._last = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params=None):
            q = (query or "").lower()
            params = params or ()
            if q.strip().startswith("insert into intellisupport.queries"):
                query_id, raw_query, intent, confidence = params
                self.store.setdefault("queries", {})[query_id] = {
                    "raw_query": raw_query,
                    "intent": intent,
                    "confidence": confidence,
                }
                self._last = None
            elif q.strip().startswith("insert into intellisupport.responses"):
                response_id, query_id, response_text, retrieved_chunk_ids = params
                self.store.setdefault("responses", {})[response_id] = {
                    "query_id": query_id,
                    "response_text": response_text,
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                }
                self._last = None
            elif "select count(*) from intellisupport.chunks" in q:
                self._last = (0,)
            elif "select query_id from intellisupport.responses where response_id" in q:
                rid = params[0]
                row = self.store.get("responses", {}).get(rid)
                self._last = (row["query_id"],) if row else None
            elif "select q.raw_query" in q and "join intellisupport.responses" in q:
                # pipeline evaluator select: return raw_query, response_text, retrieved_chunk_ids
                query_id, response_id = params
                resp = self.store.get("responses", {}).get(response_id)
                qry = self.store.get("queries", {}).get(query_id)
                if resp and qry:
                    self._last = (qry["raw_query"], resp["response_text"], resp["retrieved_chunk_ids"])
                else:
                    self._last = None
            else:
                self._last = None

        def fetchone(self):
            return self._last
        
        def fetchall(self):
            return []

    class FakeConn:
        def __init__(self):
            self.store = {}

        def cursor(self):
            return FakeCursor(self.store)

    class FakeState:
        def __init__(self):
            self.conn = FakeConn()
            self.document_loader = None
            self.chunker = None
            self.embedder = None
            self.vector_store = None
            self.bm25_retriever = None
            self.hybrid_retriever = None
            self.intent_classifier = None
            self.prompt_builder = None
            self.response_generator = None
            self.faithfulness_evaluator = None
            self.relevance_evaluator = None
            self.pipeline_evaluator = None
            self.feedback_store = None

    fake_state = FakeState()

    # Monkeypatch _ensure_app_state to return the fake state, then populate components
    import api.main as api_main

    def _fake_ensure(app):
        return fake_state

    api_main._ensure_app_state = _fake_ensure

    # Now import app and fill in component instances that rely on OpenAI but are deterministic via FakeOpenAI
    from api.main import app

    # instantiate components on fake_state using actual classes but backed by fake OpenAI
    from ingestion.loader import DocumentLoader
    from ingestion.chunker import DocumentChunker
    from ingestion.embedder import Embedder
    from retrieval.vector_store import VectorStore
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.hybrid_retriever import HybridRetriever
    from classification.intent_classifier import IntentClassifier
    from generation.prompt_builder import PromptBuilder
    from generation.response_generator import ResponseGenerator
    from evaluation.faithfulness import FaithfulnessEvaluator
    from evaluation.relevance import RelevanceEvaluator
    from evaluation.evaluator import PipelineEvaluator
    from feedback.feedback_store import FeedbackStore

    fake_state.document_loader = DocumentLoader()
    fake_state.chunker = DocumentChunker(512, 50)
    fake_state.embedder = Embedder()
    fake_state.vector_store = VectorStore(fake_state.conn)
    fake_state.bm25_retriever = BM25Retriever(fake_state.conn)
    fake_state.hybrid_retriever = HybridRetriever(fake_state.vector_store, fake_state.bm25_retriever)
    fake_state.intent_classifier = IntentClassifier()
    fake_state.prompt_builder = PromptBuilder()
    fake_state.response_generator = ResponseGenerator()
    fake_state.faithfulness_evaluator = FaithfulnessEvaluator()
    fake_state.relevance_evaluator = RelevanceEvaluator()
    fake_state.pipeline_evaluator = PipelineEvaluator(fake_state.faithfulness_evaluator, fake_state.relevance_evaluator, fake_state.conn)
    fake_state.feedback_store = FeedbackStore(fake_state.conn)

    client = TestClient(app)

    try:
        r = client.get("/health")
        assert r.status_code == 200
        # Run a deterministic query
        q = client.post("/query", json={"query": "How do I enable two-factor authentication?", "top_k": 3})
        assert q.status_code == 200
        payload = q.json()
        assert "response_id" in payload
        assert payload["intent"] == "account_management"

        # Evaluate and expect deterministic numeric fields
        ev = client.post(f"/evaluate/{payload['response_id']}")
        assert ev.status_code == 200
        evj = ev.json()
        assert "faithfulness_score" in evj
        assert evj["faithfulness_score"] >= 0.0
    finally:
        hr.HybridRetriever.retrieve_with_reranking = original
