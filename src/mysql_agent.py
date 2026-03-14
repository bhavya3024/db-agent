"""MySQL sub-agent for database operations."""

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
from src.utils import sanitize_error as _sanitize_error

load_dotenv()

MAX_RESULT_ROWS = 20
MAX_TABLES_IN_SCHEMA = 50
MAX_RESULT_CHARS = 10000


def _truncate_result(result: str) -> str:
    if len(result) > MAX_RESULT_CHARS:
        return result[:MAX_RESULT_CHARS] + f"\n\n... [truncated, showing first {MAX_RESULT_CHARS} chars]"
    return result


# ============================================================================
# DATABASE MANAGER REFERENCE
# ============================================================================

_db_manager: Optional[DatabaseManager] = None


def set_db_manager(db_manager: DatabaseManager):
    global _db_manager
    _db_manager = db_manager


def get_db_manager() -> DatabaseManager:
    if _db_manager is None:
        raise RuntimeError("Database manager not initialized. Call set_db_manager first.")
    return _db_manager


# ============================================================================
# MYSQL AGENT TOOLS
# ============================================================================

@tool
def mysql_get_schema() -> str:
    """Get the schema information for the MySQL database including all tables and their columns."""
    db_manager = get_db_manager()
    conn = db_manager.get_connection()
    if not conn:
        return "No active MySQL database connection."

    if conn.is_nosql():
        return "Current connection is MongoDB, not MySQL."

    try:
        schema_info = conn.get_schema_info()
        result = f"MySQL Database: {schema_info['database']}\n\n"

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
        return f"Error getting MySQL schema: {_sanitize_error(e)}"


@tool
def mysql_execute_query(query: str) -> str:
    """Execute a SQL query on the MySQL database.

    Args:
        query: The MySQL SQL query to execute. Can be SELECT, INSERT, UPDATE, DELETE, etc.
    """
    db_manager = get_db_manager()
    conn = db_manager.get_connection()
    if not conn:
        return "No active MySQL database connection."

    if conn.is_nosql():
        return "Current connection is MongoDB, not MySQL."

    try:
        results = conn.execute_query(query)

        if not results:
            return "Query executed successfully. No results returned."

        if "error" in results[0]:
            return f"MySQL query error: {results[0]['error']}"

        if "affected_rows" in results[0]:
            return f"Query executed successfully. Rows affected: {results[0]['affected_rows']}"

        total_results = len(results)
        display_results = results[:MAX_RESULT_ROWS] if total_results > MAX_RESULT_ROWS else results
        truncated = total_results > MAX_RESULT_ROWS

        output = json.dumps(display_results, indent=2, default=str)
        if truncated:
            output += f"\n\n... and {total_results - MAX_RESULT_ROWS} more rows (showing first {MAX_RESULT_ROWS})"

        return _truncate_result(f"Query returned {total_results} rows:\n{output}")
    except Exception as e:
        return f"Error executing MySQL query: {_sanitize_error(e)}"


@tool
def mysql_get_table_sample(table_name: str, limit: int = 5) -> str:
    """Get sample rows from a MySQL table to understand its data.

    Args:
        table_name: Name of the MySQL table to sample
        limit: Number of rows to return (default 5, max 10)
    """
    db_manager = get_db_manager()
    conn = db_manager.get_connection()
    if not conn:
        return "No active MySQL database connection."

    if conn.is_nosql():
        return "Current connection is MongoDB, not MySQL."

    try:
        results = conn.get_table_sample(table_name, min(limit, 10))
        if not results:
            return f"MySQL table '{table_name}' is empty or does not exist."

        if "error" in results[0]:
            return f"Error: {results[0]['error']}"

        return f"Sample data from MySQL table {table_name}:\n{json.dumps(results, indent=2, default=str)}"
    except Exception as e:
        return f"Error sampling MySQL table: {_sanitize_error(e)}"


# ============================================================================
# TOOL COLLECTION
# ============================================================================

mysql_tools = [mysql_get_schema, mysql_execute_query, mysql_get_table_sample]


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

MYSQL_SYSTEM_PROMPT = """You are a MySQL database expert assistant. You work exclusively with MySQL databases.

Your capabilities:
1. Get schema information using mysql_get_schema
2. Execute MySQL queries using mysql_execute_query
3. Get sample data from tables using mysql_get_table_sample

MySQL-specific guidance:
- Use backticks for identifiers: `table_name`, `column_name`
- Use LIMIT instead of FETCH FIRST
- Use SHOW TABLES, DESCRIBE, or INFORMATION_SCHEMA for metadata
- AUTO_INCREMENT (not SERIAL) for auto-increment columns
- Use IFNULL() instead of COALESCE where applicable
- String functions: CONCAT(), SUBSTRING(), LENGTH(), etc.
- Date functions: NOW(), DATE_FORMAT(), DATEDIFF(), etc.

When helping users:
- Always use mysql_get_schema first to understand the table structure
- Write MySQL-compatible queries only (not PostgreSQL or ANSI SQL that MySQL doesn't support)
- Explain your queries and results clearly
- If a query fails, check for MySQL-specific syntax issues
- Only talk about this specific MySQL connection — do not mention other database types

Always be careful with:
- Avoiding SQL injection
- Not exposing sensitive data without user consent
- Warning before executing DELETE, UPDATE, DROP, or TRUNCATE statements
"""


# ============================================================================
# LLM UTILITIES
# ============================================================================

def get_mysql_llm():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=3,
    )
    return llm.bind_tools(mysql_tools)


def invoke_llm_with_retry(llm, messages, max_retries=3):
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
# AGENT STATE
# ============================================================================

class MySQLAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    database_selected: bool
    database_type: Optional[str]
    connection_name: Optional[str]


# ============================================================================
# MYSQL AGENT NODE
# ============================================================================

def mysql_agent(state: MySQLAgentState) -> MySQLAgentState:
    """MySQL agent that handles MySQL-specific database queries."""
    llm = get_mysql_llm()
    messages = state["messages"]

    conn_name = state.get("connection_name")

    if conn_name:
        system_prompt = (
            f"You are a MySQL database assistant connected ONLY to the '{conn_name}' MySQL database.\n"
            f"You must ONLY answer questions about this specific MySQL connection. "
            f"Do NOT mention, suggest, or discuss any other database connections or types "
            f"(no PostgreSQL, no MongoDB — MySQL only).\n\n"
            + MYSQL_SYSTEM_PROMPT
        )
    else:
        db_manager = get_db_manager()
        active = db_manager.active_connection
        system_prompt = MYSQL_SYSTEM_PROMPT + f"\n\nActive MySQL database: {active}"

    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    messages_with_system = [SystemMessage(content=system_prompt)] + non_system_messages

    response = invoke_llm_with_retry(llm, messages_with_system)

    return {
        "messages": [response],
        "database_selected": state.get("database_selected", True),
        "database_type": "mysql",
    }


# ============================================================================
# TOOL NODE
# ============================================================================

def create_mysql_tools_node():
    base_tool_node = ToolNode(mysql_tools)

    def tools_with_state(state: MySQLAgentState) -> MySQLAgentState:
        result = base_tool_node.invoke(state)
        result["database_selected"] = True
        result["database_type"] = "mysql"
        result["connection_name"] = state.get("connection_name")
        return result

    return tools_with_state


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def route_after_mysql(state: MySQLAgentState) -> Literal["mysql_tools", "end"]:
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "mysql_tools"

    return "end"
