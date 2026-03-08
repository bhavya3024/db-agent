FROM langchain/langgraph-api:3.11

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml README.md /deps/

# Install dependencies
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache /deps/

# Copy application code
COPY src/ /deps/src/
COPY langgraph.json /deps/langgraph.json

ENV LANGGRAPH_CONFIG=/deps/langgraph.json

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ok')" || exit 1

# Run the LangGraph API server in production mode
CMD ["uvicorn", "langgraph_api.server:app", "--host", "0.0.0.0", "--port", "8000"]
