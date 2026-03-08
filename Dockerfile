# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency and build files
COPY pyproject.toml README.md ./

# Install dependencies using uv
RUN uv pip install --system --no-cache .

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY langgraph.json ./
COPY pyproject.toml ./

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose the LangGraph server port
EXPOSE 8000

# Set LangGraph config path
ENV LANGGRAPH_CONFIG=/app/langgraph.json
ENV LANGGRAPH_RUNTIME_EDITION=postgres

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ok')" || exit 1

# Run the LangGraph API server in production mode
CMD ["uvicorn", "langgraph_api.server:app", "--host", "0.0.0.0", "--port", "8000"]
