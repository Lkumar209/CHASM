# Contributing to CHASM

## Setup

```bash
git clone https://github.com/Lkumar209/chasm && cd chasm
uv sync --extra dev
uv run pre-commit install
```

## Commit style

[Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`.

## Branch + PR workflow

- Branch from `main`: `git checkout -b m<N>-<short-slug>`
- One PR per milestone
- CI must pass before merge (ruff, mypy, pytest, smoke e2e)

## Scientific integrity

- Never tune on `test` or `transfer` splits
- Report all cells, including failures
- All numbers come from persisted metric files — never hand-type a result

## Running checks locally

```bash
uv run ruff check src/ tests/
uv run mypy src/
uv run pytest
uv run pytest -m smoke   # smoke e2e only
```
