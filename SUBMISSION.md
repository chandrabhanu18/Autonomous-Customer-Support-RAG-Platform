# Submission Checklist and Repro Instructions

This file documents how to reproduce the acceptance tests, CI expectations, and how to configure secrets for deterministic vs. live OpenAI runs.

Quick local acceptance (Windows PowerShell):

```powershell
# Build and start service stack (Postgres + pgvector + app)
docker compose up -d --build

# Run full test suite (unit + integration)
pytest -q -r s

# Run the compose smoke test which exercises /health -> /query -> /evaluate -> /feedback -> /feedback/summary
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\scripts\compose_smoke_test.py

# Tear down after verification
docker compose down --volumes --remove-orphans
```

Expected outcomes:

- `pytest` should show the integration tests passing when the compose stack is running (example: `24 passed`).
- The smoke script should return HTTP 200 for `/health`, `/query`, `/evaluate`, `/feedback` and `/feedback/summary` and show evaluation JSON with numeric scores.
- If `OPENAI_API_KEY` is not set, the code uses deterministic local fallbacks (deterministic tests still pass).

CI / GitHub Actions notes:

- Workflows are defined in `.github/workflows`. Two main workflows exist:
  - `ci-smoke-services.yml`: service-based CI recommended for reliable integration tests.
  - `compose-smoke.yml`: compose-based CI for environments that support Docker Compose.
- To run real OpenAI-based evaluations in CI, set the repository secret `OPENAI_API_KEY`. The `ci-smoke-services.yml` workflow is stricter on `main` and will fail without a configured secret if live calls are required.

Troubleshooting:

- If tests are skipped with messages about `Database unavailable`, ensure Docker Desktop is running and `docker compose up` succeeded.
- To inspect skip reasons locally run:

```powershell
pytest -q -r s
```

Contact / Next steps:

- If you want, I can open a PR to add a release tag, enable required GitHub Actions protections, or add a `CODEOWNERS` file to help reviewers.
