"""
Discovery Agent Runner
Entry point script to run the API discovery agent
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from discovery_flow import compile_discovery_agent
from discovery_state import create_initial_state, KARACHI_LOCATION

# Load environment variables from .env file
load_dotenv()

def save_results(state, output_path="agents/langgraph_agents/discovery_agent/outputs/feature_api_mapping.json"):
    """
    Save discovery results to JSON file
    """
    # Create outputs directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare output data
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "karachi_location": KARACHI_LOCATION,
        "total_features_processed": state["total_features"],
        "feature_mappings": state["processed_features"]
    }
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {output_path}")


def print_summary(state):
    """
    Print summary of discovery results
    """
    total = state["total_features"]
    found = sum(1 for f in state["processed_features"] if f["api_found"])
    not_found = total - found
    
    print("\n" + "="*50)
    print("DISCOVERY AGENT SUMMARY")
    print("="*50)
    print(f"Total features processed: {total}")
    print(f"APIs found: {found}")
    print(f"APIs not found: {not_found}")
    print(f"Success rate: {(found/total)*100:.1f}%" if total > 0 else "0%")
    
    print("\nFeature Results:")
    for feature in state["processed_features"]:
        status = "✓" if feature["api_found"] else "✗"
        print(f"  {status} {feature['original_feature']}")
        if feature["api_found"]:
            print(f"    API: {feature['api_details'].get('name', 'Unknown')}")
        else:
            print(f"    Reason: {feature['reason']}")
    print("="*50)


def main():
    """
    Main execution function
    """
    print("Starting Discovery Agent...")
    
    # Compile the agent
    try:
        agent = compile_discovery_agent()
        print("✓ Agent compiled successfully")
    except Exception as e:
        print(f"✗ Failed to compile agent: {e}")
        return
    
    # Create initial state
    initial_state = create_initial_state([])  # Features will be loaded by feature_reader
    
    # Run the agent
    try:
        print("🔍 Running discovery process...")
        
        # Increase recursion limit and add debugging
        config = {"recursion_limit": 50}
        final_state = agent.invoke(initial_state, config=config)
        
        print("✓ Discovery completed")
        
        # Save results
        save_results(final_state)
        
        # Print summary
        print_summary(final_state)
        
    except Exception as e:
        print(f"✗ Discovery failed: {e}")
        print(f"Current state: {initial_state}")
        return


if __name__ == "__main__":
    # Ensure GEMINI_API_KEY is set
    if not os.getenv("GEMINI_API_KEY"):
        print("✗ Error: GEMINI_API_KEY environment variable not set")
        exit(1)
    
    main()