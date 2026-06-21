# Changelog

All notable changes follow [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

## [0.1.0] — M0 Scaffold

### Added
- Full repository structure: all `src/chasm/` subpackages, `configs/`, `tests/`, `docs/`
- `pyproject.toml` with `uv`, pinned extras: `local`, `api`, `probe`, `dev`
- Hydra config skeleton (`smoke` + `full` experiment configs)
- Model backends: `LocalBackend` (HF transformers), `APIBackend` (Anthropic/OpenAI), `EchoBackend` (CI/smoke)
- `ResponseCache`: disk-backed key=(model, prompt, params) cache
- `Monitor` base interface with `MonitorResult`
- Utilities: `seed_everything`, `content_hash`, `get_logger`
- `Dockerfile` + `.devcontainer/devcontainer.json`
- GitHub Actions: `ci.yml` (lint + type + test + smoke), `docs.yml`, `release.yml`
- `mkdocs.yml` with material theme; `docs/` skeleton: design, threat model, preregistration (draft), datasheet, limitations
- `LICENSE` (Apache-2.0), `.gitignore`, `.env.example`, `CODEOWNERS`, issue/PR templates, `CONTRIBUTING.md`
- `Makefile` with `ci`, `smoke`, `reproduce`, `run-all` targets
- `pre-commit` config: ruff, black, standard hooks, mypy
- Unit tests for utils, model backends, monitor base
- Smoke e2e test (`pytest -m smoke`)
