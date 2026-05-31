from __future__ import annotations

import logging
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from psycopg2.extras import Json

logger = logging.getLogger(__name__)

_DOC_ID_RE = re.compile(r"^doc_\d{3}$")


class Document(BaseModel):
    doc_id: str
    title: str
    content: str
    source_url: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, value: str) -> str:
        if not _DOC_ID_RE.match(value):
            raise ValueError("doc_id must match ^doc_\\d{3}$")
        return value

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("content cannot be empty")
        return value


class DocumentLoader:
    def load_from_dict(self, data: dict) -> Document:
        return Document(**data)

    def load_batch(self, data_list: list[dict]) -> list[Document]:
        documents: list[Document] = []
        for item in data_list:
            try:
                documents.append(self.load_from_dict(item))
            except Exception as exc:
                logger.warning("Skipping invalid document: %s", exc)
        return documents

    def save_to_db(self, documents: list[Document], conn) -> int:
        if not documents:
            return 0

        total = 0
        with conn.cursor() as cursor:
            for document in documents:
                cursor.execute(
                    """
                    INSERT INTO intellisupport.documents (doc_id, title, source_url, content, metadata, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (doc_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        source_url = EXCLUDED.source_url,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    (
                        document.doc_id,
                        document.title,
                        document.source_url,
                        document.content,
                        Json(document.metadata),
                    ),
                )
                total += cursor.rowcount
        return total
