FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_MANAGED_PYTHON=1 \
    UV_SYSTEM_CERTS=true

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project --no-editable --no-managed-python

COPY . .
RUN rm -rf .venv
RUN uv sync --frozen --no-editable --no-managed-python

FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HOST=127.0.0.1 \
    PORT=8050

WORKDIR /app

COPY --from=builder /app /app

EXPOSE 8050

CMD ["project-hack-supply-chain"]
