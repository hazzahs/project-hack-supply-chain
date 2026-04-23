FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency metadata first for better Docker cache reuse
COPY pyproject.toml uv.lock* ./

# Install dependencies, but not the project yet
RUN uv sync --frozen --no-install-project --no-editable

# Copy the application code
COPY . .

# Install the project itself into .venv
RUN uv sync --frozen --no-editable

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HOST=0.0.0.0 \
    PORT=8050

WORKDIR /app

COPY --from=builder /app /app

EXPOSE 8050

CMD ["uv", "run", "project-hack-supply-chain"]
