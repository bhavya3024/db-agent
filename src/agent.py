"""A LangGraph database agent that can interact with multiple databases."""

import os
import json
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.database import (
    DatabaseManager, 
    initialize_database_manager,
    get_postgres_config_from_env,
    DatabaseConfig,
    DatabaseType
)

# Load environment variables
load_dotenv()

# Initialize database manager at module load
db_manager = initialize_database_manager()


# Define tools for database operations
@tool
def list_databases() -> str:
    """List all available database connections."""
    connections = db_manager.list_connections()
    active = db_manager.active_connection
    result = "Available database connections:\n"
    for conn in connections:
        marker = " (active)" if conn == active else ""
        result += f"  - {conn}{marker}\n"
    return result if connections else "No database connections available."


@tool
def switch_database(database_name: str) -> str:
    """Switch to a different database connection.
    
    Args:
        database_name: Name of the database connection to switch to
    """
    if db_manager.set_active(database_name):
        return f"Switched to database: {database_name}"
    return f"Database '{database_name}' not found. Use list_databases to see available connections."


@tool
def get_schema() -> str:
    """Get the schema information for the current database including all tables and their columns."""
    conn = db_manager.get_connection()
    if not conn:
        return "No active database connection. Use list_databases to see available connections."
    
    try:
        schema_info = conn.get_schema_info()
        result = f"Database: {schema_info['database']} (Schema: {schema_info['schema']})\n\n"
        
        for table_name, table_info in schema_info["tables"].items():
            result += f"Table: {table_name}\n"
            result += "  Columns:\n"
            for col in table_info["columns"]:
                nullable = "NULL" if col["nullable"] else "NOT NULL"
                pk = " (PK)" if col["name"] in table_info["primary_keys"] else ""
                result += f"    - {col['name']}: {col['type']} {nullable}{pk}\n"
            
            if table_info["foreign_keys"]:
                result += "  Foreign Keys:\n"
                for fk in table_info["foreign_keys"]:
                    result += f"    - {fk['columns']} -> {fk['references']}\n"
            result += "\n"
        
        return result
    except Exception as e:
        return f"Error getting schema: {str(e)}"


@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query on the current database.
    
    Args:
        query: The SQL query to execute. Can be SELECT, INSERT, UPDATE, DELETE, etc.
    """
    conn = db_manager.get_connection()
    if not conn:
        return "No active database connection. Use list_databases to see available connections."
    
    try:
        results = conn.execute_query(query)
        
        if not results:
            return "Query executed successfully. No results returned."
        
        if "error" in results[0]:
            return f"Query error: {results[0]['error']}"
        
        if "affected_rows" in results[0]:
            return f"Query executed successfully. Rows affected: {results[0]['affected_rows']}"
        
        # Format results as a table
        if len(results) > 20:
            display_results = results[:20]
            truncated = True
        else:
            display_results = results
            truncated = False
        
        output = json.dumps(display_results, indent=2, default=str)
        if truncated:
            output += f"\n\n... and {len(results) - 20} more rows (showing first 20)"
        
        return f"Query returned {len(results)} rows:\n{output}"
    except Exception as e:
        return f"Error executing query: {str(e)}"


@tool
def get_table_sample(table_name: str, limit: int = 5) -> str:
    """Get sample rows from a table to understand its data.
    
    Args:
        table_name: Name of the table to sample
        limit: Number of rows to return (default 5)
    """
    conn = db_manager.get_connection()
    if not conn:
        return "No active database connection."
    
    try:
        results = conn.get_table_sample(table_name, min(limit, 10))
        if not results:
            return f"Table '{table_name}' is empty or does not exist."
        
        if "error" in results[0]:
            return f"Error: {results[0]['error']}"
        
        return f"Sample data from {table_name}:\n{json.dumps(results, indent=2, default=str)}"
    except Exception as e:
        return f"Error sampling table: {str(e)}"


# Collect all tools
tools = [list_databases, switch_database, get_schema, execute_sql, get_table_sample]


# Define the state structure
class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[list[BaseMessage], add_messages]


# System prompt for the database agent
SYSTEM_PROMPT = """You are a helpful database assistant that can interact with multiple databases.

Your capabilities:
1. List available database connections using list_databases
2. Switch between databases using switch_database
3. Get schema information using get_schema
4. Execute SQL queries using execute_sql
5. Get sample data from tables using get_table_sample

When helping users:
- First understand what database they want to work with
- Use get_schema to understand the database structure before writing queries
- Write safe, read-only queries unless explicitly asked to modify data
- Explain your queries and results clearly
- If a query fails, analyze the error and suggest corrections

Always be careful with:
- Avoiding SQL injection (use parameterized queries when possible)
- Not exposing sensitive data without user consent
- Warning before executing DELETE, UPDATE, or DROP statements
"""


# Initialize the LLM with tools
def get_llm():
    """Get the configured LLM instance with tools bound."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return llm.bind_tools(tools)


# Define nodes
def call_model(state: AgentState) -> AgentState:
    """Call the LLM with the current state."""
    llm = get_llm()
    messages = state["messages"]
    
    # Always prepend system prompt for the LLM call
    # Filter out any existing system messages to avoid duplicates
    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + non_system_messages
    
    response = llm.invoke(messages_with_system)
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if we should continue with tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM wants to use tools, route to tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end
    return "end"


# Build the graph
def create_graph():
    """Create and compile the LangGraph state graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # After tools, go back to agent
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


def run_agent(user_input: str):
    """Run the agent with a user input."""
    graph = create_graph()
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=user_input)]
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    # Get the final response
    final_message = result["messages"][-1]
    return final_message.content


if __name__ == "__main__":
    # Example usage
    print("Database Agent Ready!")
    print("-" * 50)
    
    # Test listing databases
    response = run_agent("What databases are available and what tables do they have?")
    print(f"Agent Response:\n{response}")

