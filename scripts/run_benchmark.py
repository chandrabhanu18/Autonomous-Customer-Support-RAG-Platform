from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification.intent_classifier import IntentClassifier
from config import settings
from db import create_connection
from evaluation.evaluator import PipelineEvaluator
from evaluation.faithfulness import FaithfulnessEvaluator
from evaluation.relevance import RelevanceEvaluator
from ingestion.chunker import DocumentChunker
from ingestion.embedder import Embedder
from ingestion.loader import DocumentLoader
from ingestion.seed_data import SEED_DOCUMENTS
from tests.test_evaluation import BENCHMARK_TEST_CASES


def main() -> None:
    conn = create_connection()
    try:
        migration_path = Path(__file__).resolve().parents[1] / "database" / "migrations" / "001_initial.sql"
        migration_sql = migration_path.read_text(encoding="utf-8")
        with conn.cursor() as cursor:
            cursor.execute(migration_sql)
            cursor.execute(
                """
                TRUNCATE TABLE
                    intellisupport.feedback,
                    intellisupport.responses,
                    intellisupport.queries,
                    intellisupport.chunks,
                    intellisupport.documents
                RESTART IDENTITY CASCADE
                """
            )

        loader = DocumentLoader()
        chunker = DocumentChunker(settings.chunk_size, settings.chunk_overlap)
        embedder = Embedder(settings.embedding_model)
        documents = loader.load_batch(SEED_DOCUMENTS)
        loader.save_to_db(documents, conn)
        chunks = chunker.chunk_batch(documents)
        embedder.embed_and_store_chunks(chunks, conn)

        evaluator = PipelineEvaluator(
            FaithfulnessEvaluator(settings.generation_model),
            RelevanceEvaluator(settings.generation_model),
            conn,
        )
        report = evaluator.run_benchmark(BENCHMARK_TEST_CASES)
        print(json.dumps(report.model_dump(), indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
