"""A LangGraph database agent with PostgreSQL and MongoDB sub-agents."""

import os
import time
from typing import TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from openai import RateLimitError

from src.database import (
    DatabaseManager, 
    initialize_database_manager,
)

# Import sub-agents
from src.postgres_agent import (
    postgres_agent,
    postgres_tools,
    create_postgres_tools_node,
    route_after_postgres,
    set_db_manager as set_postgres_db_manager,
)
from src.mongodb_agent import (
    mongodb_agent,
    mongodb_tools,
    create_mongodb_tools_node,
    route_after_mongodb,
    set_db_manager as set_mongodb_db_manager,
)

# Load environment variables
load_dotenv()

# Initialize database manager at module load
db_manager = initialize_database_manager()

# Share database manager with sub-agents
set_postgres_db_manager(db_manager)
set_mongodb_db_manager(db_manager)


# ============================================================================
# ROUTER AGENT TOOLS
# ============================================================================

@tool
def list_databases() -> str:
    """List all available database connections."""
    connections = db_manager.list_connections()
    active = db_manager.active_connection
    result = "Available database connections:\n"
    for conn in connections:
        conn_obj = db_manager.get_connection(conn)
        db_type = "MongoDB" if conn_obj and conn_obj.is_nosql() else "PostgreSQL"
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
        conn = db_manager.get_connection(database_name)
        db_type = "mongodb" if conn and conn.is_nosql() else "postgres"
        return f"Switched to database: {database_name} (type: {db_type})"
    return f"Database '{database_name}' not found. Use list_databases to see available connections."


# Router tools
router_tools = [list_databases, switch_database]


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """The state of the main agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    database_selected: bool
    database_type: Optional[str]  # "postgres", "mongodb", or None


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

ROUTER_SYSTEM_PROMPT = """You are a helpful database assistant that can interact with PostgreSQL and MongoDB databases.

Your role is to:
1. List available database connections using list_databases
2. Help users switch to their desired database using switch_database

When a user starts a conversation:
- First, list the available databases
- Ask which database they want to work with
- Use switch_database to connect to their chosen database

Once a database is selected, the appropriate specialized agent (PostgreSQL or MongoDB) will handle the queries.
"""


# ============================================================================
# LLM UTILITIES
# ============================================================================

def get_router_llm():
    """Get the configured LLM instance with router tools bound."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=3,
    )
    return llm.bind_tools(router_tools)


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


def _check_database_selected_in_messages(messages: list[BaseMessage]) -> tuple[bool, Optional[str]]:
    """Check if switch_database was called in the conversation history.
    
    Returns:
        Tuple of (database_selected, database_type)
    """
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if "Switched to database:" in content:
                if "type: mongodb" in content:
                    return True, "mongodb"
                elif "type: postgres" in content:
                    return True, "postgres"
                return True, None
    return False, None


# ============================================================================
# ROUTER AGENT (Main Agent)
# ============================================================================

def router_agent(state: AgentState) -> AgentState:
    """Router agent that handles database selection."""
    llm = get_router_llm()
    messages = state["messages"]
    
    # Check current database status
    database_selected = state.get("database_selected", False)
    database_type = state.get("database_type")
    
    if not database_selected:
        database_selected, database_type = _check_database_selected_in_messages(messages)
    
    # Build system prompt with context
    connections = db_manager.list_connections()
    if connections:
        db_list = ", ".join(f"{c} [{'MongoDB' if db_manager.get_connection(c).is_nosql() else 'PostgreSQL'}]" for c in connections)
        system_prompt = ROUTER_SYSTEM_PROMPT + f"\n\nAvailable databases: {db_list}"
    else:
        system_prompt = ROUTER_SYSTEM_PROMPT + "\n\nNo database connections are available."
    
    # Add current status
    active = db_manager.active_connection
    if active:
        conn = db_manager.get_connection()
        db_type_str = "MongoDB" if conn and conn.is_nosql() else "PostgreSQL"
        system_prompt += f"\n\nCurrent active database: {active} ({db_type_str})"
    
    # Prepare messages
    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    messages_with_system = [SystemMessage(content=system_prompt)] + non_system_messages
    
    response = invoke_llm_with_retry(llm, messages_with_system)
    
    return {
        "messages": [response],
        "database_selected": database_selected,
        "database_type": database_type
    }


# ============================================================================
# TOOL NODES
# ============================================================================

def create_router_tools_node():
    """Create tools node for the router agent."""
    base_tool_node = ToolNode(router_tools)
    
    def tools_with_state_update(state: AgentState) -> AgentState:
        result = base_tool_node.invoke(state)
        
        database_selected = state.get("database_selected", False)
        database_type = state.get("database_type")
        
        # Check if switch_database was called successfully
        for msg in result.get("messages", []):
            if isinstance(msg, ToolMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if "Switched to database:" in content:
                    database_selected = True
                    if "type: mongodb" in content:
                        database_type = "mongodb"
                    elif "type: postgres" in content:
                        database_type = "postgres"
                    break
        
        result["database_selected"] = database_selected
        result["database_type"] = database_type
        return result
    
    return tools_with_state_update


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def route_after_router(state: AgentState) -> Literal["router_tools", "postgres_agent", "mongodb_agent", "end"]:
    """Determine next step after the router agent."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the router wants to use tools (list/switch database)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "router_tools"
    
    # Check if database is selected and route to appropriate sub-agent
    database_selected = state.get("database_selected", False)
    database_type = state.get("database_type")
    
    if database_selected and database_type:
        if database_type == "postgres":
            return "postgres_agent"
        elif database_type == "mongodb":
            return "mongodb_agent"
    
    # If no database selected yet, end and wait for user input
    return "end"


def route_after_router_tools(state: AgentState) -> Literal["router_agent", "postgres_agent", "mongodb_agent"]:
    """Determine next step after router tools execution."""
    database_selected = state.get("database_selected", False)
    database_type = state.get("database_type")
    
    # If database is now selected, route to appropriate sub-agent
    if database_selected and database_type:
        if database_type == "postgres":
            return "postgres_agent"
        elif database_type == "mongodb":
            return "mongodb_agent"
    
    # Otherwise, go back to router
    return "router_agent"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_graph():
    """Create and compile the LangGraph state graph with sub-agents."""
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("router_agent", router_agent)
    workflow.add_node("router_tools", create_router_tools_node())
    workflow.add_node("postgres_agent", postgres_agent)
    workflow.add_node("postgres_tools", create_postgres_tools_node())
    workflow.add_node("mongodb_agent", mongodb_agent)
    workflow.add_node("mongodb_tools", create_mongodb_tools_node())
    
    # Set entry point - always start with router
    workflow.set_entry_point("router_agent")
    
    # Router agent routing
    workflow.add_conditional_edges(
        "router_agent",
        route_after_router,
        {
            "router_tools": "router_tools",
            "postgres_agent": "postgres_agent",
            "mongodb_agent": "mongodb_agent",
            "end": END
        }
    )
    
    # After router tools, decide where to go
    workflow.add_conditional_edges(
        "router_tools",
        route_after_router_tools,
        {
            "router_agent": "router_agent",
            "postgres_agent": "postgres_agent",
            "mongodb_agent": "mongodb_agent"
        }
    )
    
    # PostgreSQL agent routing
    workflow.add_conditional_edges(
        "postgres_agent",
        route_after_postgres,
        {
            "postgres_tools": "postgres_tools",
            "end": END
        }
    )
    
    # After postgres tools, go back to postgres agent
    workflow.add_edge("postgres_tools", "postgres_agent")
    
    # MongoDB agent routing
    workflow.add_conditional_edges(
        "mongodb_agent",
        route_after_mongodb,
        {
            "mongodb_tools": "mongodb_tools",
            "end": END
        }
    )
    
    # After mongodb tools, go back to mongodb agent
    workflow.add_edge("mongodb_tools", "mongodb_agent")
    
    return workflow.compile()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_agent(user_input: str, database_type: Optional[str] = None):
    """Run the agent with a user input.
    
    Args:
        user_input: The user's message
        database_type: Optional pre-selected database type ("postgres" or "mongodb")
    """
    graph = create_graph()
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "database_selected": database_type is not None,
        "database_type": database_type
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    # Get the final response
    final_message = result["messages"][-1]
    return final_message.content


if __name__ == "__main__":
    # Example usage
    print("Database Agent Ready!")
    print("=" * 50)
    print("This agent has two sub-agents:")
    print("  - PostgreSQL Agent: For SQL database queries")
    print("  - MongoDB Agent: For NoSQL database queries")
    print("=" * 50)
    
    # Test the agent
    response = run_agent("Hello, I want to query some data.")
    print(f"Agent Response:\n{response}")
