# DB Agent - LangGraph Boilerplate

A LangGraph-based agent application built with Python and managed with `uv`.

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

1. Install uv (if not already installed):
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Clone the repository and navigate to the project directory

3. Install dependencies using uv:
```bash
uv sync
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running the Agent

Activate the virtual environment and run the agent:

```bash
# Activate the environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows

# Run the agent
python src/agent.py
```

Alternatively, use uv to run directly:

```bash
uv run python src/agent.py
```

### Using in Your Code

```python
from src.agent import run_agent

response = run_agent("What is LangGraph?")
print(response)
```

## Project Structure

```
db-agent/
├── src/
│   ├── __init__.py
│   └── agent.py          # Main agent implementation
├── .env.example          # Environment variables template
├── .gitignore
├── pyproject.toml        # Project configuration and dependencies
├── README.md
└── LICENSE
```

## LangGraph Basics

This boilerplate demonstrates:

- **State Management**: Using `TypedDict` and `Annotated` for type-safe state
- **Graph Building**: Creating nodes and edges with `StateGraph`
- **Message Handling**: Managing conversation history with `add_messages`
- **Conditional Logic**: Using `add_conditional_edges` for flow control

## Adding New Nodes

To extend the agent with new capabilities:

```python
def my_custom_node(state: AgentState) -> AgentState:
    # Your logic here
    return {"messages": [...]}

# Add to graph
workflow.add_node("custom", my_custom_node)
workflow.add_edge("agent", "custom")
```

## Development

Install development dependencies:

```bash
uv sync --all-extras
```

This includes Jupyter for interactive development.

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [LangChain Documentation](https://python.langchain.com/)

## License

See LICENSE file for details.
