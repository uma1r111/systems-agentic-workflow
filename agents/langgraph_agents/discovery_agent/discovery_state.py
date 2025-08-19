"""
Discovery Agent State Schema for LangGraph
Handles state management for API discovery and validation workflow with feature normalization
"""

from typing import TypedDict, List, Dict, Optional, Any
from datetime import datetime


class DiscoveryState(TypedDict):
    """
    State schema for the Discovery Agent that handles feature reading,
    feature normalization, API discovery, and API validation for AQI prediction features.
    """
    
    # Current feature being processed
    feature: str
    
    # Feature normalization fields
    normalization_status: str  # "checking" | "already_exists" | "missing" | "error"
    mapped_to: Optional[str]   # Existing column name if feature already exists
    existing_columns: List[str] # CSV columns for semantic comparison
    similarity_scores: Dict[str, float] # Column name -> similarity score
    normalization_confidence: float # Confidence in the mapping (0.0-1.0)
    
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
    status: str  # "reading" | "normalizing" | "searching" | "found" | "not_found" | "validating" | "valid" | "invalid" | "completed" | "error"
    result_status: str  # "found" | "not_found" | "already_exists" | "searching"
    
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
    
    # CSV data source configuration
    csv_file_path: Optional[str]  # Path to existing data CSV
    csv_columns_loaded: bool  # Flag to track if CSV columns have been loaded


# Constants for Karachi location
KARACHI_LOCATION = {
    "lat": 24.8607,
    "lon": 67.0011,
    "city": "Karachi",
    "country": "Pakistan"
}

# Default values for state initialization
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_SIMILARITY_THRESHOLD = 0.7  # Threshold for considering features as similar

def create_initial_state(features_to_add: List[str], csv_file_path: Optional[str] = None) -> DiscoveryState:
    """
    Create initial state for the discovery agent
    
    Args:
        features_to_add: List of features from drift monitoring agent
        csv_file_path: Optional path to existing CSV file for column comparison
        
    Returns:
        DiscoveryState: Initialized state object
    """
    return DiscoveryState(
        feature="",
        
        # Feature normalization initialization
        normalization_status="checking",
        mapped_to=None,
        existing_columns=[],
        similarity_scores={},
        normalization_confidence=0.0,
        
        # Search configuration
        search_attempts=0,
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        search_keywords=[],
        
        # Location
        karachi_coords=KARACHI_LOCATION,
        
        # API discovery
        candidate_api=None,
        api_candidates=[],
        validated_api=None,
        
        # Status tracking
        status="reading",
        result_status="searching",
        confidence_score=0.0,
        reason=None,
        suggested_alternatives=[],
        
        # Processing metadata
        timestamp=datetime.now().isoformat(),
        current_feature_index=0,
        total_features=len(features_to_add),
        features_to_process=features_to_add,
        processed_features=[],
        
        # CSV configuration
        csv_file_path=csv_file_path,
        csv_columns_loaded=False
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
        
        # Reset normalization fields
        "normalization_status": "checking",
        "mapped_to": None,
        "similarity_scores": {},
        "normalization_confidence": 0.0,
        
        # Reset search fields
        "search_attempts": 0,
        "search_keywords": [],
        "candidate_api": None,
        "api_candidates": [],
        "validated_api": None,
        
        # Reset status
        "status": "normalizing",
        "result_status": "searching",
        "confidence_score": 0.0,
        "reason": None,
        "suggested_alternatives": []
    })
    return state


def update_normalization_result(state: DiscoveryState, 
                               normalization_status: str,
                               mapped_to: Optional[str] = None,
                               similarity_scores: Optional[Dict[str, float]] = None,
                               confidence: float = 0.0,
                               reason: Optional[str] = None) -> DiscoveryState:
    """
    Update state with feature normalization results
    
    Args:
        state: Current state
        normalization_status: Result of normalization check
        mapped_to: Existing column name if feature already exists
        similarity_scores: Similarity scores for all columns
        confidence: Confidence in the mapping
        reason: Additional context or reasoning
        
    Returns:
        Updated state
    """
    state.update({
        "normalization_status": normalization_status,
        "mapped_to": mapped_to,
        "similarity_scores": similarity_scores or {},
        "normalization_confidence": confidence,
        "reason": reason
    })
    
    # Update result status based on normalization
    if normalization_status == "already_exists":
        state["result_status"] = "already_exists"
        state["status"] = "completed"  # Skip API search
    elif normalization_status == "missing":
        state["result_status"] = "searching"
        state["status"] = "searching"  # Continue to API search
    else:
        state["status"] = "error"
        
    return state


def load_csv_columns(state: DiscoveryState, columns: List[str]) -> DiscoveryState:
    """
    Load CSV columns into state for comparison
    
    Args:
        state: Current state
        columns: List of column names from CSV
        
    Returns:
        Updated state with CSV columns loaded
    """
    state["existing_columns"] = columns
    state["csv_columns_loaded"] = True
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
    Add current feature to processed features list with enhanced metadata
    
    Args:
        state: Current state
        
    Returns:
        Updated state with feature added to processed list
    """
    processed_feature = {
        "original_feature": state["feature"],
        "normalization_status": state["normalization_status"],
        "mapped_to": state["mapped_to"],
        "similarity_scores": state["similarity_scores"],
        "normalization_confidence": state["normalization_confidence"],
        "search_keywords": state["search_keywords"],
        "api_found": state["result_status"] == "found",
        "api_details": state["validated_api"] if state["validated_api"] else {},
        "confidence_score": state["confidence_score"],
        "reason": state["reason"],
        "suggested_alternatives": state["suggested_alternatives"],
        "search_attempts": state["search_attempts"],
        "final_status": state["result_status"]
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


def is_feature_already_exists(state: DiscoveryState) -> bool:
    """
    Check if current feature already exists in CSV
    
    Args:
        state: Current state
        
    Returns:
        True if feature already exists, False otherwise
    """
    return state["normalization_status"] == "already_exists"


def should_skip_api_search(state: DiscoveryState) -> bool:
    """
    Determine if API search should be skipped based on normalization results
    
    Args:
        state: Current state
        
    Returns:
        True if API search should be skipped, False otherwise
    """
    return (state["normalization_status"] == "already_exists" or 
            state["status"] == "completed" or
            state["status"] == "error")


def get_similarity_threshold() -> float:
    """
    Get the similarity threshold for feature matching
    
    Returns:
        Similarity threshold value
    """
    return DEFAULT_SIMILARITY_THRESHOLD


def get_processing_summary(state: DiscoveryState) -> Dict[str, Any]:
    """
    Get a summary of processing results
    
    Args:
        state: Current state
        
    Returns:
        Dictionary with processing summary
    """
    total = state["total_features"]
    already_exists = sum(1 for f in state["processed_features"] if f["final_status"] == "already_exists")
    api_found = sum(1 for f in state["processed_features"] if f["final_status"] == "found")
    not_found = total - already_exists - api_found
    
    return {
        "total_features": total,
        "already_exists": already_exists,
        "new_apis_found": api_found,
        "not_found": not_found,
        "already_exists_rate": (already_exists/total)*100 if total > 0 else 0,
        "api_discovery_rate": (api_found/total)*100 if total > 0 else 0,
        "overall_success_rate": ((already_exists + api_found)/total)*100 if total > 0 else 0
    }