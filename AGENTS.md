# Repository Guidelines

## Project Structure & Module Organization
- `openhands/`: Python backend (server, runtime, agenthub, prompts).
- `frontend/`: React/Remix SPA; `frontend/src` for UI code and `frontend/public` for static assets.
- `openhands-ui/`: reusable React component library.
- `openhands-cli/`: CLI package.
- `tests/`: `unit/`, `runtime/`, and `e2e/` suites with per-suite README guidance.
- `evaluation/`: benchmark and analysis tooling.
- `enterprise/`: enterprise-only features under a separate license.
- `containers/`: Docker images and dev-container setup.

## Build, Test, and Development Commands
- `make build`: install Python and frontend dependencies, set up hooks, and build the frontend.
- `make setup-config`: create `config.toml` with LLM settings.
- `make run`: start backend + frontend together; use `make start-backend` or `make start-frontend` to run them separately.
- `make lint`: run backend pre-commit checks and frontend linting.
- `make test`: run frontend tests; backend tests are run with pytest (see Testing).
- `make docker-dev` / `make docker-run`: use the dev container or run the app via Docker Compose.

## Coding Style & Naming Conventions
- Python formatting and linting are enforced via pre-commit; `ruff` rules live in `dev_config/python/ruff.toml` and `black` settings in `pyproject.toml` (single-quote strings, double-quote docstrings).
- Frontend linting is via `npm run lint` in `frontend/`.
- Follow existing naming patterns: Python tests use `test_*.py`, frontend tests use `*.test.tsx`, and component files are typically kebab-case under `frontend/src/components`.

## Testing Guidelines
- Backend unit tests: `poetry run pytest ./tests/unit` (see `tests/unit/README.md`).
- E2E tests: `cd tests/e2e` then run pytest with Playwright (requires `GITHUB_TOKEN`, `LLM_MODEL`, `LLM_API_KEY`); use `--base-url` to target a running instance.
- Frontend tests: `cd frontend && npm run test` or `npm run test:coverage` (Vitest + React Testing Library).

## Commit & Pull Request Guidelines
- Recent git history uses conventional commit-style prefixes like `feat`, `fix`, and `chore` with optional scopes; follow this style for commits and PR titles.
- PR titles should match the conventional prefixes listed in `CONTRIBUTING.md` (e.g., `feat(frontend): ...`).
- PR descriptions can be brief for small changes; include a changelog note for user-facing updates.

## Configuration & Secrets
- Use `make setup-config` or `config.template.toml` as a starting point; keep API keys out of git.
- Frontend env vars live in `frontend/.env` (copy from `frontend/.env.sample`).
