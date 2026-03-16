FROM langchain/langgraph-api:3.12-wolfi

# Copy application code into the deps directory
ADD src /deps/__outer_src/src
ADD pyproject.toml README.md /deps/__outer_src/

# Install dependencies using the base image's constraints to avoid version conflicts
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/__outer_src

# Tell the LangGraph runtime where to find the graph
ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_src/src/agent.py:create_graph"}'
ENV PORT=8000
