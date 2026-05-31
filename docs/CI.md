# CI and Secrets

This repository includes CI workflows to run a lightweight smoke test that exercises the FastAPI endpoints.

Workflows:
- `.github/workflows/compose-smoke.yml` — runs `docker compose up` and executes the smoke test (useful for local parity).
- `.github/workflows/ci-smoke-services.yml` — recommended for CI (uses GitHub Actions service container for Postgres and runs the app on the runner).

Secrets:
- `OPENAI_API_KEY` — Optional but recommended. If not set, the app falls back to deterministic behavior for evaluation and tests (see code comments).

CI behavior:
- On branches and pull requests the `ci-smoke-services.yml` workflow will warn if `OPENAI_API_KEY` is missing and continue running with deterministic fallbacks.
- On `main` the workflow is stricter: it will fail the job early if `OPENAI_API_KEY` is not set. This prevents accidental merges that would make the main branch's CI results nondeterministic.

How to add the secret (GitHub):
1. Go to your repository on GitHub.
2. Settings → Secrets and variables → Actions → New repository secret.
3. Name: `OPENAI_API_KEY`, Value: your OpenAI API key.

Local testing:
1. Export the env var locally (PowerShell):
   ```powershell
   $env:OPENAI_API_KEY = 'sk-...'
   ```
2. Run the local smoke test via the Makefile:
   ```bash
   make smoke
   ```

If you prefer CI to fail when the secret is not set, update `.github/workflows/ci-smoke-services.yml` accordingly.
