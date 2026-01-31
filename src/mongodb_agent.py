"""MongoDB sub-agent for database operations."""

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
MAX_COLLECTIONS_IN_SCHEMA = 50
MAX_RESULT_CHARS = 10000


def _truncate_result(result: str) -> str:
    """Truncate result string if too long."""
    if len(result) > MAX_RESULT_CHARS:
        return result[:MAX_RESULT_CHARS] + f"\n\n... [truncated, showing first {MAX_RESULT_CHARS} chars]"
    return result


# ============================================================================
# MONGODB AGENT TOOLS
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
def mongodb_get_schema() -> str:
    """Get the schema information for the MongoDB database including all collections and their fields."""
    db_manager = get_db_manager()
    conn = db_manager.get_connection()
    if not conn:
        return "No active database connection."
    
    if not conn.is_nosql():
        return "Current database is PostgreSQL, not MongoDB."
    
    try:
        schema_info = conn.get_schema_info()
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
                result += f"  Fields (showing {len(fields)} of {len(coll_info['fields'])}):\n"
                for field in fields:
                    result += f"    - {field['name']}: {field['type']}\n"
            result += "\n"
        
        return _truncate_result(result)
    except Exception as e:
        return f"Error getting schema: {str(e)}"


@tool
def mongodb_execute_query(query: str) -> str:
    """Execute a MongoDB query on the current database.
    
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
    db_manager = get_db_manager()
    conn = db_manager.get_connection()
    if not conn:
        return "No active database connection."
    
    if not conn.is_nosql():
        return "Current database is PostgreSQL, not MongoDB."
    
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
def mongodb_get_collection_sample(collection_name: str, limit: int = 5) -> str:
    """Get sample documents from a MongoDB collection to understand its data.
    
    Args:
        collection_name: Name of the collection to sample
        limit: Number of documents to return (default 5)
    """
    db_manager = get_db_manager()
    conn = db_manager.get_connection()
    if not conn:
        return "No active database connection."
    
    if not conn.is_nosql():
        return "Current database is PostgreSQL, not MongoDB."
    
    try:
        results = conn.get_table_sample(collection_name, min(limit, 10))
        if not results:
            return f"Collection '{collection_name}' is empty or does not exist."
        
        if "error" in results[0]:
            return f"Error: {results[0]['error']}"
        
        return f"Sample data from {collection_name}:\n{json.dumps(results, indent=2, default=str)}"
    except Exception as e:
        return f"Error sampling collection: {str(e)}"


# ============================================================================
# TOOL COLLECTION
# ============================================================================

mongodb_tools = [mongodb_get_schema, mongodb_execute_query, mongodb_get_collection_sample]


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

MONGODB_SYSTEM_PROMPT = """You are a MongoDB database expert assistant.

Your capabilities:
1. Get schema information using mongodb_get_schema
2. Execute MongoDB queries using mongodb_execute_query
3. Get sample data from collections using mongodb_get_collection_sample

When helping users:
- Use mongodb_get_schema to understand the database structure before writing queries
- Write queries as JSON objects with the operation, collection, and parameters
- Write safe, read-only queries unless explicitly asked to modify data
- Explain your queries and results clearly
- If a query fails, analyze the error and suggest corrections

MongoDB Query Examples:
- Find all users: {"operation": "find", "collection": "users", "filter": {}, "limit": 10}
- Find by field: {"operation": "find", "collection": "users", "filter": {"age": {"$gt": 25}}}
- Count documents: {"operation": "count", "collection": "users", "filter": {}}
- Aggregate: {"operation": "aggregate", "collection": "orders", "pipeline": [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]}

Always be careful with:
- Not exposing sensitive data without user consent
- Warning before executing delete or update operations
"""


# ============================================================================
# LLM UTILITIES
# ============================================================================

def get_mongodb_llm():
    """Get the configured LLM instance with MongoDB tools bound."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=3,
    )
    return llm.bind_tools(mongodb_tools)


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

class MongoDBAgentState(TypedDict):
    """The state of the MongoDB agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    database_selected: bool
    database_type: Optional[str]


# ============================================================================
# MONGODB AGENT NODE
# ============================================================================

def mongodb_agent(state: MongoDBAgentState) -> MongoDBAgentState:
    """MongoDB agent that handles NoSQL database queries."""
    db_manager = get_db_manager()
    llm = get_mongodb_llm()
    messages = state["messages"]
    
    # Build system prompt with context
    active = db_manager.active_connection
    system_prompt = MONGODB_SYSTEM_PROMPT + f"\n\nActive database: {active}"
    
    # Prepare messages
    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    messages_with_system = [SystemMessage(content=system_prompt)] + non_system_messages
    
    response = invoke_llm_with_retry(llm, messages_with_system)
    
    return {
        "messages": [response],
        "database_selected": state.get("database_selected", True),
        "database_type": "mongodb"
    }


# ============================================================================
# TOOL NODE
# ============================================================================

def create_mongodb_tools_node():
    """Create tools node for the MongoDB agent."""
    base_tool_node = ToolNode(mongodb_tools)
    
    def tools_with_state(state: MongoDBAgentState) -> MongoDBAgentState:
        result = base_tool_node.invoke(state)
        result["database_selected"] = True
        result["database_type"] = "mongodb"
        return result
    
    return tools_with_state


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def route_after_mongodb(state: MongoDBAgentState) -> Literal["mongodb_tools", "end"]:
    """Determine next step after MongoDB agent."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "mongodb_tools"
    
    return "end"
