.PHONY: install lint fmt type test smoke clean docs serve-docs docker-build reproduce run-all

install:
	uv sync --extra dev

lint:
	uv run ruff check src/ tests/

fmt:
	uv run black src/ tests/
	uv run ruff check --fix src/ tests/

type:
	uv run mypy src/

test:
	uv run pytest

smoke:
	uv run pytest -m smoke -v

test-det:
	uv run pytest -m determinism -v

ci: lint type test smoke

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage coverage.xml htmlcov/

docs:
	uv run mkdocs build --strict

serve-docs:
	uv run mkdocs serve

docker-build:
	docker build -t chasm:latest .

reproduce:
	bash scripts/reproduce.sh

run-all:
	bash scripts/run_all.sh
