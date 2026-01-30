"""A simple LangGraph agent example with state management."""

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()


# Define the state structure
class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[list[BaseMessage], add_messages]


# Initialize the LLM
def get_llm():
    """Get the configured LLM instance."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )


# Define nodes
def call_model(state: AgentState) -> AgentState:
    """Call the LLM with the current state."""
    llm = get_llm()
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message is from the AI, end
    if isinstance(last_message, AIMessage):
        return "end"
    return "continue"


# Build the graph
def create_graph():
    """Create and compile the LangGraph state graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "agent",
            "end": END
        }
    )
    
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
    response = run_agent("What is LangGraph and how does it work?")
    print(f"Agent Response: {response}")
