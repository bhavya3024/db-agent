"""A LangGraph database agent that can interact with multiple databases."""

import os
import json
import time
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from openai import RateLimitError

from src.database import (
    DatabaseManager, 
    initialize_database_manager,
    get_postgres_config_from_env,
    get_mongodb_config_from_env,
    DatabaseConfig,
    DatabaseType
)

# Load environment variables
load_dotenv()

# Constants for limiting result sizes to avoid token limits
MAX_RESULT_ROWS = 20
MAX_TABLES_IN_SCHEMA = 50
MAX_COLLECTIONS_IN_SCHEMA = 50
MAX_RESULT_CHARS = 10000  # Truncate large results

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
        conn_obj = db_manager.get_connection(conn)
        db_type = "MongoDB" if conn_obj and conn_obj.is_nosql() else "SQL"
        marker = " (active)" if conn == active else ""
        result += f"  - {conn} [{db_type}]{marker}\n"
    
    if not connections:
        return "No database connections available."
    
    return result


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
    """Get the schema information for the current database including all tables/collections and their columns/fields."""
    conn = db_manager.get_connection()
    if not conn:
        return "No active database connection. Use list_databases to see available connections."
    
    try:
        schema_info = conn.get_schema_info()
        
        # Handle MongoDB schema
        if schema_info.get("type") == "mongodb":
            result = f"Database: {schema_info['database']} (MongoDB)\n\n"
            
            collections = list(schema_info.get("collections", {}).items())
            if len(collections) > MAX_COLLECTIONS_IN_SCHEMA:
                result += f"Note: Showing first {MAX_COLLECTIONS_IN_SCHEMA} of {len(collections)} collections\n\n"
                collections = collections[:MAX_COLLECTIONS_IN_SCHEMA]
            
            for coll_name, coll_info in collections:
                result += f"Collection: {coll_name}\n"
                result += f"  Documents: {coll_info.get('document_count', 0)}\n"
                
                if coll_info.get("fields"):
                    fields = coll_info["fields"][:20]  # Limit fields shown
                    result += f"  Fields (showing {len(fields)} of {len(coll_info['fields'])}):"  + "\n"
                    for field in fields:
                        result += f"    - {field['name']}: {field['type']}\n"
                result += "\n"
            
            return _truncate_result(result)
        
        # Handle SQL database schema
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


def _truncate_result(result: str) -> str:
    """Truncate result string if too long."""
    if len(result) > MAX_RESULT_CHARS:
        return result[:MAX_RESULT_CHARS] + f"\n\n... [truncated, showing first {MAX_RESULT_CHARS} chars]"
    return result


@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query on the current database. Use this for PostgreSQL, MySQL, or SQLite databases.
    
    Args:
        query: The SQL query to execute. Can be SELECT, INSERT, UPDATE, DELETE, etc.
    """
    conn = db_manager.get_connection()
    if not conn:
        return "No active database connection. Use list_databases to see available connections."
    
    if conn.is_nosql():
        return "Current database is MongoDB. Use execute_mongodb instead of execute_sql."
    
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
def execute_mongodb(query: str) -> str:
    """Execute a MongoDB query on the current database. Use this for MongoDB databases.
    
    Args:
        query: A JSON string describing the MongoDB operation. Supported operations:
            - Find: {"operation": "find", "collection": "users", "filter": {"age": {"$gt": 25}}, "limit": 10}
            - Find One: {"operation": "find_one", "collection": "users", "filter": {"_id": "..."}}
            - Insert One: {"operation": "insert_one", "collection": "users", "document": {"name": "John"}}
            - Insert Many: {"operation": "insert_many", "collection": "users", "documents": [{...}, {...}]}
            - Update One: {"operation": "update_one", "collection": "users", "filter": {...}, "update": {"$set": {...}}}
            - Update Many: {"operation": "update_many", "collection": "users", "filter": {...}, "update": {"$set": {...}}}
            - Delete One: {"operation": "delete_one", "collection": "users", "filter": {...}}
            - Delete Many: {"operation": "delete_many", "collection": "users", "filter": {...}}
            - Aggregate: {"operation": "aggregate", "collection": "users", "pipeline": [{"$match": {...}}, {"$group": {...}}]}
            - Count: {"operation": "count", "collection": "users", "filter": {}}
    """
    conn = db_manager.get_connection()
    if not conn:
        return "No active database connection. Use list_databases to see available connections."
    
    if not conn.is_nosql():
        return "Current database is SQL-based. Use execute_sql instead of execute_mongodb."
    
    try:
        results = conn.execute_query(query)
        
        if not results:
            return "Query executed successfully. No results returned."
        
        if results and "error" in results[0]:
            return f"Query error: {results[0]['error']}"
        
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
            output += f"\n\n... and {total_results - MAX_RESULT_ROWS} more documents (showing first {MAX_RESULT_ROWS})"
        
        return _truncate_result(f"Query returned {total_results} document(s):\n{output}")
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
tools = [list_databases, switch_database, get_schema, execute_sql, execute_mongodb, get_table_sample]


# Define the state structure
class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    database_selected: bool  # Track if user has selected a database in this thread


# System prompt for the database agent
SYSTEM_PROMPT = """You are a helpful database assistant that can interact with multiple databases including SQL databases (PostgreSQL, MySQL, SQLite) and NoSQL databases (MongoDB).

Your capabilities:
1. List available database connections using list_databases
2. Switch between databases using switch_database
3. Get schema information using get_schema (works for both SQL tables and MongoDB collections)
4. Execute SQL queries using execute_sql (for PostgreSQL, MySQL, SQLite)
5. Execute MongoDB queries using execute_mongodb (for MongoDB)
6. Get sample data from tables/collections using get_table_sample

When helping users:
- Use get_schema to understand the database structure before writing queries
- For SQL databases, write standard SQL queries
- For MongoDB, write queries as JSON objects with the operation, collection, and parameters
- Write safe, read-only queries unless explicitly asked to modify data
- Explain your queries and results clearly
- If a query fails, analyze the error and suggest corrections

MongoDB Query Examples:
- Find all users: {"operation": "find", "collection": "users", "filter": {}, "limit": 10}
- Find by field: {"operation": "find", "collection": "users", "filter": {"age": {"$gt": 25}}}
- Count documents: {"operation": "count", "collection": "users", "filter": {}}
- Aggregate: {"operation": "aggregate", "collection": "orders", "pipeline": [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]}

Always be careful with:
- Avoiding injection attacks
- Not exposing sensitive data without user consent
- Warning before executing DELETE, UPDATE, or DROP/remove statements
"""


# Initialize the LLM with tools
def get_llm():
    """Get the configured LLM instance with tools bound."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=3,  # Built-in retry for rate limits
    )
    return llm.bind_tools(tools)


def invoke_llm_with_retry(llm, messages, max_retries=3):
    """Invoke LLM with exponential backoff retry for rate limits."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
            print(f"Rate limit hit, waiting {wait_time}s before retry...")
            time.sleep(wait_time)
        except Exception as e:
            # Check if it's a rate limit error wrapped in another exception
            if "429" in str(e) or "rate_limit" in str(e).lower():
                if attempt == max_retries - 1:
                    raise
                wait_time = (2 ** attempt) * 5
                print(f"Rate limit hit, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise


def _check_database_selected_in_messages(messages: list[BaseMessage]) -> bool:
    """Check if switch_database was called in the conversation history."""
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if "Switched to database:" in content:
                return True
    return False


# Define nodes
def call_model(state: AgentState) -> AgentState:
    """Call the LLM with the current state."""
    llm = get_llm()
    messages = state["messages"]
    
    # Check if database has been selected (from state or message history)
    database_selected = state.get("database_selected", False)
    if not database_selected:
        database_selected = _check_database_selected_in_messages(messages)
    
    # Build context-aware system prompt
    system_prompt = SYSTEM_PROMPT
    
    # Check if this is a new thread (first human message)
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    is_new_thread = len(human_messages) == 1
    
    # Add database selection status to context
    connections = db_manager.list_connections()
    active = db_manager.active_connection
    conn = db_manager.get_connection()
    db_type = "MongoDB" if conn and conn.is_nosql() else "SQL"
    
    if is_new_thread and not database_selected:
        # New thread - ask user to select database
        if connections:
            db_list = ", ".join(f"{c} [{'MongoDB' if db_manager.get_connection(c).is_nosql() else 'SQL'}]" for c in connections)
            system_prompt += f"\n\n**IMPORTANT**: This is a new conversation. Before proceeding, you MUST list the available databases and ask the user which one they want to work with. Available databases: {db_list}."
        else:
            system_prompt += "\n\n**Current Status**: No database connections are available. Please inform the user that no databases are configured."
    else:
        # Existing thread or database already selected
        system_prompt += f"\n\n**Current Status**: Active database is '{active}' ({db_type})."
    
    # Always prepend system prompt for the LLM call
    # Filter out any existing system messages to avoid duplicates
    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    messages_with_system = [SystemMessage(content=system_prompt)] + non_system_messages
    
    # Use retry logic for rate limits
    response = invoke_llm_with_retry(llm, messages_with_system)
    
    # Update state with database selection status
    return {"messages": [response], "database_selected": database_selected}


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
def create_tools_node():
    """Create a tools node that also tracks database selection."""
    base_tool_node = ToolNode(tools)
    
    def tools_with_state_update(state: AgentState) -> AgentState:
        # Run the base tool node
        result = base_tool_node.invoke(state)
        
        # Check if switch_database was called successfully
        database_selected = state.get("database_selected", False)
        if not database_selected:
            for msg in result.get("messages", []):
                if isinstance(msg, ToolMessage):
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if "Switched to database:" in content:
                        database_selected = True
                        break
        
        result["database_selected"] = database_selected
        return result
    
    return tools_with_state_update


def create_graph():
    """Create and compile the LangGraph state graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", create_tools_node())
    
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
        "messages": [HumanMessage(content=user_input)],
        "database_selected": False
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
    
    # Test the agent - it should ask for database selection first
    response = run_agent("Hello, I want to query some data.")
    print(f"Agent Response:\n{response}")

