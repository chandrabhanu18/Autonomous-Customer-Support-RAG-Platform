from __future__ import annotations

import pytest

from ingestion.chunker import DocumentChunker
from ingestion.embedder import EmbeddingError
from ingestion.loader import DocumentLoader
from ingestion.seed_data import SEED_DOCUMENTS


def test_load_from_dict_valid():
    loader = DocumentLoader()
    document = loader.load_from_dict(SEED_DOCUMENTS[0])
    assert document.doc_id == SEED_DOCUMENTS[0]["doc_id"]
    assert document.title == SEED_DOCUMENTS[0]["title"]
    assert document.content == SEED_DOCUMENTS[0]["content"]
    assert document.source_url == SEED_DOCUMENTS[0]["source_url"]


def test_load_from_dict_invalid_doc_id():
    loader = DocumentLoader()
    payload = {**SEED_DOCUMENTS[0], "doc_id": "document_1"}
    with pytest.raises(ValueError):
        loader.load_from_dict(payload)


def test_load_from_dict_empty_content():
    loader = DocumentLoader()
    payload = {**SEED_DOCUMENTS[0], "content": "   "}
    with pytest.raises(ValueError):
        loader.load_from_dict(payload)


def test_chunk_document_chunk_ids():
    loader = DocumentLoader()
    document = loader.load_from_dict(SEED_DOCUMENTS[0])
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_document(document)
    assert chunks
    assert all(chunk.chunk_id.startswith("chunk_doc_001_") for chunk in chunks)


def test_chunk_overlap():
    loader = DocumentLoader()
    document = loader.load_from_dict(
        {
            "doc_id": "doc_999",
            "title": "Overlap Test",
            "content": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen",
        }
    )
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=3)
    chunks = chunker.chunk_document(document)
    assert len(chunks) >= 2
    first_tokens = chunks[0].content.split()
    second_tokens = chunks[1].content.split()
    assert first_tokens[-3:] == second_tokens[:3]


def test_embed_text_shape(services):
    embedding = services["embedder"].embed_text("How do I connect Slack?")
    assert len(embedding) == 1536
