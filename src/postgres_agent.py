"""PostgreSQL sub-agent for database operations."""

import os
import json
import time
from typing import TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from openai import RateLimitError

from src.database import DatabaseManager

# Load environment variables
load_dotenv()

# Constants for limiting result sizes
MAX_RESULT_ROWS = 20
MAX_TABLES_IN_SCHEMA = 50
MAX_RESULT_CHARS = 10000


def _truncate_result(result: str) -> str:
    """Truncate result string if too long."""
    if len(result) > MAX_RESULT_CHARS:
        return result[:MAX_RESULT_CHARS] + f"\n\n... [truncated, showing first {MAX_RESULT_CHARS} chars]"
    return result


# ============================================================================
# POSTGRESQL AGENT TOOLS
# ============================================================================

# Database manager reference (set by main agent)
_db_manager: Optional[DatabaseManager] = None


def set_db_manager(db_manager: DatabaseManager):
    """Set the database manager reference."""
    global _db_manager
    _db_manager = db_manager


def get_db_manager() -> DatabaseManager:
    """Get the database manager reference."""
    if _db_manager is None:
        raise RuntimeError("Database manager not initialized. Call set_db_manager first.")
    return _db_manager


@tool
def postgres_get_schema() -> str:
    """Get the schema information for the PostgreSQL database including all tables and their columns."""
    db_manager = get_db_manager()
    conn = db_manager.get_connection()
    if not conn:
        return "No active database connection."
    
    if conn.is_nosql():
        return "Current database is MongoDB, not PostgreSQL."
    
    try:
        schema_info = conn.get_schema_info()
        result = f"Database: {schema_info['database']} (Schema: {schema_info['schema']})\n\n"
        
        tables = list(schema_info["tables"].items())
        if len(tables) > MAX_TABLES_IN_SCHEMA:
            result += f"Note: Showing first {MAX_TABLES_IN_SCHEMA} of {len(tables)} tables\n\n"
            tables = tables[:MAX_TABLES_IN_SCHEMA]
        
        for table_name, table_info in tables:
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
        
        return _truncate_result(result)
    except Exception as e:
        return f"Error getting schema: {str(e)}"


@tool
def postgres_execute_sql(query: str) -> str:
    """Execute a SQL query on the PostgreSQL database.
    
    Args:
        query: The SQL query to execute. Can be SELECT, INSERT, UPDATE, DELETE, etc.
    """
    db_manager = get_db_manager()
    conn = db_manager.get_connection()
    if not conn:
        return "No active database connection."
    
    if conn.is_nosql():
        return "Current database is MongoDB, not PostgreSQL."
    
    try:
        results = conn.execute_query(query)
        
        if not results:
            return "Query executed successfully. No results returned."
        
        if "error" in results[0]:
            return f"Query error: {results[0]['error']}"
        
        if "affected_rows" in results[0]:
            return f"Query executed successfully. Rows affected: {results[0]['affected_rows']}"
        
        # Limit results to avoid token limits
        total_results = len(results)
        if total_results > MAX_RESULT_ROWS:
            display_results = results[:MAX_RESULT_ROWS]
            truncated = True
        else:
            display_results = results
            truncated = False
        
        output = json.dumps(display_results, indent=2, default=str)
        if truncated:
            output += f"\n\n... and {total_results - MAX_RESULT_ROWS} more rows (showing first {MAX_RESULT_ROWS})"
        
        return _truncate_result(f"Query returned {total_results} rows:\n{output}")
    except Exception as e:
        return f"Error executing query: {str(e)}"


@tool
def postgres_get_table_sample(table_name: str, limit: int = 5) -> str:
    """Get sample rows from a PostgreSQL table to understand its data.
    
    Args:
        table_name: Name of the table to sample
        limit: Number of rows to return (default 5)
    """
    db_manager = get_db_manager()
    conn = db_manager.get_connection()
    if not conn:
        return "No active database connection."
    
    if conn.is_nosql():
        return "Current database is MongoDB, not PostgreSQL."
    
    try:
        results = conn.get_table_sample(table_name, min(limit, 10))
        if not results:
            return f"Table '{table_name}' is empty or does not exist."
        
        if "error" in results[0]:
            return f"Error: {results[0]['error']}"
        
        return f"Sample data from {table_name}:\n{json.dumps(results, indent=2, default=str)}"
    except Exception as e:
        return f"Error sampling table: {str(e)}"


# ============================================================================
# TOOL COLLECTION
# ============================================================================

postgres_tools = [postgres_get_schema, postgres_execute_sql, postgres_get_table_sample]


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

POSTGRES_SYSTEM_PROMPT = """You are a PostgreSQL database expert assistant.

Your capabilities:
1. Get schema information using postgres_get_schema
2. Execute SQL queries using postgres_execute_sql
3. Get sample data from tables using postgres_get_table_sample

When helping users:
- Use postgres_get_schema to understand the database structure before writing queries
- Write standard PostgreSQL queries
- Write safe, read-only queries unless explicitly asked to modify data
- Explain your queries and results clearly
- If a query fails, analyze the error and suggest corrections

Always be careful with:
- Avoiding SQL injection
- Not exposing sensitive data without user consent
- Warning before executing DELETE, UPDATE, or DROP statements
"""


# ============================================================================
# LLM UTILITIES
# ============================================================================

def get_postgres_llm():
    """Get the configured LLM instance with PostgreSQL tools bound."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=3,
    )
    return llm.bind_tools(postgres_tools)


def invoke_llm_with_retry(llm, messages, max_retries=3):
    """Invoke LLM with exponential backoff retry for rate limits."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt) * 5
            print(f"Rate limit hit, waiting {wait_time}s before retry...")
            time.sleep(wait_time)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                if attempt == max_retries - 1:
                    raise
                wait_time = (2 ** attempt) * 5
                print(f"Rate limit hit, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise


# ============================================================================
# AGENT STATE (shared with main agent)
# ============================================================================

class PostgresAgentState(TypedDict):
    """The state of the PostgreSQL agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    database_selected: bool
    database_type: Optional[str]


# ============================================================================
# POSTGRESQL AGENT NODE
# ============================================================================

def postgres_agent(state: PostgresAgentState) -> PostgresAgentState:
    """PostgreSQL agent that handles SQL database queries."""
    db_manager = get_db_manager()
    llm = get_postgres_llm()
    messages = state["messages"]
    
    # Build system prompt with context
    active = db_manager.active_connection
    system_prompt = POSTGRES_SYSTEM_PROMPT + f"\n\nActive database: {active}"
    
    # Prepare messages
    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    messages_with_system = [SystemMessage(content=system_prompt)] + non_system_messages
    
    response = invoke_llm_with_retry(llm, messages_with_system)
    
    return {
        "messages": [response],
        "database_selected": state.get("database_selected", True),
        "database_type": "postgres"
    }


# ============================================================================
# TOOL NODE
# ============================================================================

def create_postgres_tools_node():
    """Create tools node for the PostgreSQL agent."""
    base_tool_node = ToolNode(postgres_tools)
    
    def tools_with_state(state: PostgresAgentState) -> PostgresAgentState:
        result = base_tool_node.invoke(state)
        result["database_selected"] = True
        result["database_type"] = "postgres"
        return result
    
    return tools_with_state


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def route_after_postgres(state: PostgresAgentState) -> Literal["postgres_tools", "end"]:
    """Determine next step after PostgreSQL agent."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "postgres_tools"
    
    return "end"
