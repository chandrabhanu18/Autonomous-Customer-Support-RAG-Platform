from __future__ import annotations

import time
from typing import Optional

from openai import OpenAI
from psycopg2.extras import Json

from config import settings
from db import vector_to_pgvector_literal
from ingestion.chunker import Chunk
from utils import deterministic_embedding


class EmbeddingError(RuntimeError):
    pass


class Embedder:
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 100):
        self.model = model
        self.batch_size = batch_size
        self._client: Optional[OpenAI] = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def _embed_with_client(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.embeddings.create(model=self.model, input=texts)
                return [list(item.embedding) for item in response.data]
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
        raise EmbeddingError(str(last_error) if last_error else "Embedding API call failed")

    def embed_text(self, text: str) -> list[float]:
        if not self._client:
            return deterministic_embedding(text)
        embeddings = self._embed_with_client([text])
        if not embeddings:
            raise EmbeddingError("No embedding returned")
        embedding = embeddings[0]
        if len(embedding) != 1536:
            raise EmbeddingError("Unexpected embedding size")
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self._client:
            return [deterministic_embedding(text) for text in texts]
        embeddings: list[list[float]] = []
        for index in range(0, len(texts), self.batch_size):
            batch = texts[index:index + self.batch_size]
            embeddings.extend(self._embed_with_client(batch))
        return embeddings

    def embed_and_store_chunks(self, chunks: list[Chunk], conn) -> int:
        if not chunks:
            return 0
        embeddings = self.embed_batch([chunk.content for chunk in chunks])
        count = 0
        with conn.cursor() as cursor:
            for chunk, embedding in zip(chunks, embeddings):
                cursor.execute(
                    """
                    INSERT INTO intellisupport.chunks
                        (chunk_id, doc_id, content, chunk_index, token_count, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        doc_id = EXCLUDED.doc_id,
                        content = EXCLUDED.content,
                        chunk_index = EXCLUDED.chunk_index,
                        token_count = EXCLUDED.token_count,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.content,
                        chunk.chunk_index,
                        chunk.token_count,
                        vector_to_pgvector_literal(embedding),
                        Json(chunk.metadata),
                    ),
                )
                count += cursor.rowcount
        return count
