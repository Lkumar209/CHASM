FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1

WORKDIR /workspace

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.11 /uv /usr/local/bin/uv

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install Python deps (no GPU extras by default; override with --build-arg EXTRAS=local)
ARG EXTRAS=dev
RUN uv sync --extra ${EXTRAS}

COPY . .

# Default: run tests
CMD ["uv", "run", "pytest", "-m", "smoke"]
