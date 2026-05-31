#!/bin/sh
set -eu

python /app/scripts/run_migrations.py
python /app/scripts/seed_database.py

exec "$@"
