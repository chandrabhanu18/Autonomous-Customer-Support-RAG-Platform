from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from db import create_connection
from ingestion.chunker import DocumentChunker
from ingestion.embedder import Embedder
from ingestion.loader import DocumentLoader
from ingestion.seed_data import SEED_DOCUMENTS


def main() -> None:
    conn = create_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM intellisupport.documents")
            count = int(cursor.fetchone()[0])
            if count > 0:
                print(f"Seed skipped: {count} documents already present")
                return

        loader = DocumentLoader()
        chunker = DocumentChunker(settings.chunk_size, settings.chunk_overlap)
        embedder = Embedder(settings.embedding_model)
        documents = loader.load_batch(SEED_DOCUMENTS)
        loader.save_to_db(documents, conn)
        chunks = chunker.chunk_batch(documents)
        embedder.embed_and_store_chunks(chunks, conn)
        print(f"Seeded {len(documents)} documents and {len(chunks)} chunks")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
