from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import create_connection


def main() -> None:
    conn = create_connection()
    try:
        migration_path = ROOT / "database" / "migrations" / "001_initial.sql"
        migration_sql = migration_path.read_text(encoding="utf-8")
        with conn.cursor() as cursor:
            cursor.execute(migration_sql)
        print("Migrations applied")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
