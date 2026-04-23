FROM ghcr.io/astral-sh/uv:debian-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON=3.12

WORKDIR /app

RUN uv python install 3.12

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

FROM ghcr.io/astral-sh/uv:debian-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:/root/.local/bin:$PATH" \
    HOST=0.0.0.0 \
    PORT=8050 \
    UV_PYTHON=3.12 \
    DEBUG=0

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

EXPOSE 8050

CMD ["uv", "run", "project-hack-supply-chain"]
