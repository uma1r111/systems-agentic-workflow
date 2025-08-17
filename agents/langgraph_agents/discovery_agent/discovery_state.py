"""
Discovery Agent State Schema for LangGraph
Handles state management for API discovery and validation workflow
"""

from typing import TypedDict, List, Dict, Optional, Any
from datetime import datetime


class DiscoveryState(TypedDict):
    """
    State schema for the Discovery Agent that handles feature reading,
    API discovery, and API validation for AQI prediction features.
    """
    
    # Current feature being processed
    feature: str
    
    # Search configuration
    search_attempts: int
    max_attempts: int
    search_keywords: List[str]
    
    # Geographic constraints (fixed for Karachi)
    karachi_coords: Dict[str, float]
    
    # API discovery results
    candidate_api: Optional[Dict[str, Any]]
    api_candidates: List[Dict[str, Any]]  # Track all found options
    
    # Validation results
    validated_api: Optional[Dict[str, Any]]
    
    # Processing status
    status: str  # "searching" | "found" | "not_found" | "validating" | "valid" | "invalid"
    result_status: str  # "found" | "not_found" | "searching"
    
    # Confidence and reasoning
    confidence_score: float
    reason: Optional[str]  # Reason for failure or additional context
    suggested_alternatives: List[str]
    
    # Processing metadata
    timestamp: str
    current_feature_index: int
    total_features: int
    
    # All features from drift analysis
    features_to_process: List[str]
    processed_features: List[Dict[str, Any]]


# Constants for Karachi location
KARACHI_LOCATION = {
    "lat": 24.8607,
    "lon": 67.0011,
    "city": "Karachi",
    "country": "Pakistan"
}

# Default values for state initialization
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

def create_initial_state(features_to_add: List[str]) -> DiscoveryState:
    """
    Create initial state for the discovery agent
    
    Args:
        features_to_add: List of features from drift monitoring agent
        
    Returns:
        DiscoveryState: Initialized state object
    """
    return DiscoveryState(
        feature="",
        search_attempts=0,
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        search_keywords=[],
        karachi_coords=KARACHI_LOCATION,
        candidate_api=None,
        api_candidates=[],
        validated_api=None,
        status="searching",
        result_status="searching",
        confidence_score=0.0,
        reason=None,
        suggested_alternatives=[],
        timestamp=datetime.now().isoformat(),
        current_feature_index=0,
        total_features=len(features_to_add),
        features_to_process=features_to_add,
        processed_features=[]
    )


def update_feature_processing(state: DiscoveryState, feature: str) -> DiscoveryState:
    """
    Update state for processing a new feature
    
    Args:
        state: Current state
        feature: Feature to process
        
    Returns:
        Updated state
    """
    state.update({
        "feature": feature,
        "search_attempts": 0,
        "search_keywords": [],
        "candidate_api": None,
        "api_candidates": [],
        "validated_api": None,
        "status": "searching",
        "result_status": "searching",
        "confidence_score": 0.0,
        "reason": None,
        "suggested_alternatives": []
    })
    return state


def increment_search_attempt(state: DiscoveryState) -> DiscoveryState:
    """
    Increment search attempt counter
    
    Args:
        state: Current state
        
    Returns:
        Updated state with incremented attempt count
    """
    state["search_attempts"] += 1
    return state


def has_exceeded_max_attempts(state: DiscoveryState) -> bool:
    """
    Check if maximum search attempts have been exceeded
    
    Args:
        state: Current state
        
    Returns:
        True if max attempts exceeded, False otherwise
    """
    return state["search_attempts"] >= state["max_attempts"]


def add_processed_feature(state: DiscoveryState) -> DiscoveryState:
    """
    Add current feature to processed features list
    
    Args:
        state: Current state
        
    Returns:
        Updated state with feature added to processed list
    """
    processed_feature = {
        "original_feature": state["feature"],
        "search_keywords": state["search_keywords"],
        "api_found": state["result_status"] == "found",
        "api_details": state["validated_api"] if state["validated_api"] else {},
        "confidence_score": state["confidence_score"],
        "reason": state["reason"],
        "suggested_alternatives": state["suggested_alternatives"],
        "search_attempts": state["search_attempts"]
    }
    
    state["processed_features"].append(processed_feature)
    state["current_feature_index"] += 1
    
    return state


def has_more_features(state: DiscoveryState) -> bool:
    """
    Check if there are more features to process
    
    Args:
        state: Current state
        
    Returns:
        True if more features to process, False otherwise
    """
    return state["current_feature_index"] < state["total_features"]


def get_next_feature(state: DiscoveryState) -> Optional[str]:
    """
    Get the next feature to process
    
    Args:
        state: Current state
        
    Returns:
        Next feature string or None if no more features
    """
    if has_more_features(state):
        return state["features_to_process"][state["current_feature_index"]]
    return None