# ─── Stage 1: dependency installer ───────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy only dependency manifest first — layer cached until pyproject.toml changes
COPY pyproject.toml uv.lock ./

# Install production dependencies (no project source yet, no dev extras)
RUN uv sync --frozen --no-dev --no-install-project

# ─── Stage 2: final image ─────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy uv binary from builder
COPY --from=builder /bin/uv /bin/uvx /bin/

# Copy installed site-packages from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source
COPY . .

# Make the venv the default Python
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# SQLite checkpoint file lives in a mount-able volume
VOLUME ["/app/data"]
ENV SQLITE_DB_PATH=/app/data/checkpoints.db

# Default: run the Telegram bot
CMD ["python", "examples/telegram_bot.py"]
