"""
Discovery Agent Flow for LangGraph
Defines the graph structure and conditional logic for API discovery with feature normalization
"""

from langgraph.graph import StateGraph, END
from discovery_state import (
    DiscoveryState, 
    has_exceeded_max_attempts, 
    has_more_features, 
    add_processed_feature,
    should_skip_api_search,
    is_feature_already_exists
)
from discovery_nodes import feature_reader, feature_normalizer, api_searcher, api_validator


def should_continue_after_feature_read(state: DiscoveryState) -> str:
    """
    Check what to do after reading a feature
    """
    if state["status"] == "completed":
        return "end"
    elif state["status"] == "error":
        return "end"
    else:
        return "normalize"


def should_continue_after_normalization(state: DiscoveryState) -> str:
    """
    Conditional logic after feature normalization
    Determines whether to skip API search or continue
    """
    if state["status"] == "error":
        return "process_next"
    elif state["normalization_status"] == "already_exists":
        # Feature already exists, skip API search
        return "process_next"
    elif state["normalization_status"] == "missing":
        # Feature is missing, proceed to API search
        return "search"
    else:
        # Fallback for unexpected states
        return "process_next"


def should_retry_search(state: DiscoveryState) -> str:
    """
    Conditional logic for retry mechanism after API validation
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
    print(f"📝 Processing completed for feature: {state['feature']}")
    print(f"   Status: {state.get('normalization_status', 'unknown')} | Result: {state.get('result_status', 'unknown')}")
    
    # Add feature to processed list with all metadata
    state = add_processed_feature(state)
    
    # Log summary for this feature
    current_feature = state["processed_features"][-1] if state["processed_features"] else {}
    if current_feature.get("normalization_status") == "already_exists":
        print(f"   ✅ Feature already exists as: {current_feature.get('mapped_to')}")
    elif current_feature.get("api_found"):
        print(f"   ✅ API found: {current_feature.get('api_details', {}).get('name', 'Unknown')}")
    else:
        print(f"   ❌ No solution found: {current_feature.get('reason', 'Unknown reason')}")
    
    return state


def build_discovery_graph() -> StateGraph:
    """
    Build and return the enhanced discovery agent graph with feature normalization
    """
    # Initialize graph
    graph = StateGraph(DiscoveryState)
    
    # Add nodes
    graph.add_node("feature_reader", feature_reader)
    graph.add_node("feature_normalizer", feature_normalizer)
    graph.add_node("api_searcher", api_searcher)
    graph.add_node("api_validator", api_validator)
    graph.add_node("process_next", process_next_feature)
    
    # Add direct edges (non-conditional)
    graph.add_edge("api_searcher", "api_validator")
    
    # Add conditional edges
    
    # After reading feature: either end or normalize
    graph.add_conditional_edges(
        "feature_reader",
        should_continue_after_feature_read,
        {
            "normalize": "feature_normalizer",
            "end": END
        }
    )
    
    # After normalization: either search for API or process next feature
    graph.add_conditional_edges(
        "feature_normalizer",
        should_continue_after_normalization,
        {
            "search": "api_searcher",
            "process_next": "process_next"
        }
    )
    
    # After API validation: either retry search or process next
    graph.add_conditional_edges(
        "api_validator",
        should_retry_search,
        {
            "retry_search": "api_searcher",
            "process_next": "process_next"
        }
    )
    
    # After processing feature: either continue with next feature or end
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


def validate_graph_structure():
    """
    Validate the graph structure and flow logic
    Returns True if valid, False otherwise
    """
    try:
        # Test compilation
        graph = build_discovery_graph()
        compiled_graph = graph.compile()
        
        print("✅ Graph structure validation passed")
        print("\nGraph Flow Summary:")
        print("1. feature_reader → feature_normalizer")
        print("2. feature_normalizer → [api_searcher OR process_next]")
        print("3. api_searcher → api_validator")
        print("4. api_validator → [retry api_searcher OR process_next]")
        print("5. process_next → [continue to feature_reader OR END]")
        
        return True
        
    except Exception as e:
        print(f"❌ Graph validation failed: {e}")
        return False


def get_flow_description():
    """
    Return a description of the complete workflow
    """
    return """
    Enhanced Discovery Agent Workflow:
    
    1. FEATURE READING
       - Load features from drift analysis
       - Process features one by one
       
    2. FEATURE NORMALIZATION
       - Check if feature already exists in CSV under different name
       - Use LLM for semantic comparison
       - Decision: already_exists → skip API search, missing → continue
       
    3. API DISCOVERY (only if feature missing)
       - Search web using Tavily + LLM
       - Find candidate APIs for the feature
       - Multiple search strategies and fallbacks
       
    4. API VALIDATION (only if candidate found)
       - Test API accessibility and response
       - Validate URL format and endpoints
       - Retry mechanism for failed validations
       
    5. FEATURE PROCESSING
       - Record results (normalization + API discovery)
       - Track metadata and confidence scores
       - Continue to next feature or end
       
    Key Benefits:
    - Avoids duplicate data collection
    - Real web search for API discovery
    - Comprehensive error handling
    - Detailed result tracking
    """


if __name__ == "__main__":
    # Validate graph when run directly
    print("Validating Discovery Agent Graph Structure...")
    if validate_graph_structure():
        print("\n" + get_flow_description())
    else:
        print("Graph validation failed - check node definitions and imports")