from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from ingestion.loader import Document


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    token_count: int
    metadata: dict = Field(default_factory=dict)


class DocumentChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))

    def chunk_document(self, document: Document) -> list[Chunk]:
        tokens = document.content.split()
        if not tokens:
            return []

        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0
        while start < len(tokens):
            end = min(len(tokens), start + self.chunk_size)
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                break
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{document.doc_id}_{chunk_index}",
                    doc_id=document.doc_id,
                    content=" ".join(chunk_tokens),
                    chunk_index=chunk_index,
                    token_count=len(chunk_tokens),
                    metadata={"title": document.title, **document.metadata},
                )
            )
            if end >= len(tokens):
                break
            start += step
            chunk_index += 1
        return chunks

    def chunk_batch(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks
