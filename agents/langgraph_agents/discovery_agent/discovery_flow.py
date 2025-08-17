"""
Discovery Agent Flow for LangGraph
Defines the graph structure and conditional logic for API discovery
"""

from langgraph.graph import StateGraph, END
from discovery_state import DiscoveryState, has_exceeded_max_attempts, has_more_features, add_processed_feature
from discovery_nodes import feature_reader, api_searcher, api_validator


def should_retry_search(state: DiscoveryState) -> str:
    """
    Conditional logic for retry mechanism
    """
    if state["status"] == "valid":
        return "process_next"
    elif state["status"] == "not_found":
        return "process_next"  # Don't retry if no API found
    elif state["status"] == "error":
        return "process_next"  # Don't retry on errors
    elif state["status"] == "invalid" and not has_exceeded_max_attempts(state):
        return "retry_search"
    else:
        return "process_next"


def should_continue_after_feature_read(state: DiscoveryState) -> str:
    """
    Check what to do after reading a feature
    """
    if state["status"] == "completed":
        return "end"
    elif state["status"] == "error":
        return "end"
    else:
        return "search"


def should_continue_processing(state: DiscoveryState) -> str:
    """
    Conditional logic for processing next feature
    """
    if has_more_features(state):
        return "continue"
    else:
        return "end"


def process_next_feature(state: DiscoveryState) -> DiscoveryState:
    """
    Add current feature to processed list and continue
    """
    state = add_processed_feature(state)
    return state


def build_discovery_graph() -> StateGraph:
    """
    Build and return the discovery agent graph
    """
    # Initialize graph
    graph = StateGraph(DiscoveryState)
    
    # Add nodes
    graph.add_node("feature_reader", feature_reader)
    graph.add_node("api_searcher", api_searcher)
    graph.add_node("api_validator", api_validator)
    graph.add_node("process_next", process_next_feature)
    
    # Add edges
    graph.add_edge("api_searcher", "api_validator")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "feature_reader",
        should_continue_after_feature_read,
        {
            "search": "api_searcher",
            "end": END
        }
    )
    
    # Add conditional edges
    graph.add_conditional_edges(
        "api_validator",
        should_retry_search,
        {
            "retry_search": "api_searcher",
            "process_next": "process_next"
        }
    )
    
    graph.add_conditional_edges(
        "process_next",
        should_continue_processing,
        {
            "continue": "feature_reader",
            "end": END
        }
    )
    
    # Set entry point
    graph.set_entry_point("feature_reader")
    
    return graph


def compile_discovery_agent():
    """
    Compile and return the discovery agent
    """
    graph = build_discovery_graph()
    return graph.compile()