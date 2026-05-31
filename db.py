from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg2

from config import settings


@contextmanager
def get_connection() -> Iterator[psycopg2.extensions.connection]:
    conn = psycopg2.connect(settings.database_url)
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()


def create_connection() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(settings.database_url)
    conn.autocommit = True
    return conn


def vector_to_pgvector_literal(values: list[float]) -> str:
    return "[" + ", ".join(f"{float(value):.8f}" for value in values) + "]"
