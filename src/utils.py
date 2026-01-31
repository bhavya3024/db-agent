"""Shared utilities for database agents."""

import os
import time
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from openai import RateLimitError


# Constants for limiting result sizes to avoid token limits
MAX_RESULT_ROWS = 20
MAX_TABLES_IN_SCHEMA = 50
MAX_COLLECTIONS_IN_SCHEMA = 50
MAX_RESULT_CHARS = 10000


class AgentState(TypedDict):
    """The state of the main agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    database_selected: bool
    database_type: Optional[str]  # "postgres", "mongodb", or None


def truncate_result(result: str) -> str:
    """Truncate result string if too long."""
    if len(result) > MAX_RESULT_CHARS:
        return result[:MAX_RESULT_CHARS] + f"\n\n... [truncated, showing first {MAX_RESULT_CHARS} chars]"
    return result


def get_llm(tools: list):
    """Get the configured LLM instance with specified tools bound."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=3,
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


def check_database_selected_in_messages(messages: list[BaseMessage]) -> tuple[bool, Optional[str]]:
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
