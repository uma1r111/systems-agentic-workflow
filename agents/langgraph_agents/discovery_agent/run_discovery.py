"""
Discovery Agent Runner
Entry point script to run the enhanced API discovery agent with feature normalization
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from discovery_flow import compile_discovery_agent
from discovery_state import create_initial_state, KARACHI_LOCATION, get_processing_summary

# Load environment variables from .env file
load_dotenv()

def save_results(state, output_path="agents/langgraph_agents/discovery_agent/outputs/feature_api_mapping.json"):
    """
    Save discovery results to JSON file with enhanced metadata
    """
    # Create outputs directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get processing summary
    summary = get_processing_summary(state)
    
    # Prepare output data with enhanced structure
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "karachi_location": KARACHI_LOCATION,
            "total_features_processed": state["total_features"],
            "csv_file_analyzed": state.get("csv_file_path", "data/full_preprocessed_aqi_weather_data_with_all_features.csv"),
            "csv_columns_count": len(state.get("existing_columns", [])),
            "processing_summary": summary
        },
        "results": {
            "features_already_existing": [],
            "new_apis_discovered": [],
            "features_not_found": []
        },
        "detailed_feature_mappings": state["processed_features"]
    }
    
    # Categorize results for easier consumption
    for feature in state["processed_features"]:
        feature_info = {
            "original_feature": feature["original_feature"],
            "confidence": feature.get("normalization_confidence", feature.get("confidence_score", 0.0)),
            "reasoning": feature.get("reason", "")
        }
        
        if feature["normalization_status"] == "already_exists":
            feature_info["mapped_to"] = feature["mapped_to"]
            feature_info["similarity_scores"] = feature.get("similarity_scores", {})
            output_data["results"]["features_already_existing"].append(feature_info)
            
        elif feature["api_found"]:
            feature_info["api_details"] = feature["api_details"]
            feature_info["search_attempts"] = feature["search_attempts"]
            output_data["results"]["new_apis_discovered"].append(feature_info)
            
        else:
            feature_info["search_attempts"] = feature.get("search_attempts", 0)
            feature_info["suggested_alternatives"] = feature.get("suggested_alternatives", [])
            output_data["results"]["features_not_found"].append(feature_info)
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"📁 Results saved to {output_path}")
    return output_path


def print_enhanced_summary(state):
    """
    Print comprehensive summary of discovery results
    """
    summary = get_processing_summary(state)
    
    print("\n" + "="*60)
    print("🔍 ENHANCED DISCOVERY AGENT SUMMARY")
    print("="*60)
    print(f"📊 Total features processed: {summary['total_features']}")
    print(f"✅ Features already existing: {summary['already_exists']} ({summary['already_exists_rate']:.1f}%)")
    print(f"🆕 New APIs discovered: {summary['new_apis_found']} ({summary['api_discovery_rate']:.1f}%)")
    print(f"❌ Features not found: {summary['not_found']}")
    print(f"🎯 Overall success rate: {summary['overall_success_rate']:.1f}%")
    
    print("\n📋 DETAILED RESULTS:")
    print("-" * 40)
    
    # Show already existing features
    if summary['already_exists'] > 0:
        print(f"\n✅ FEATURES ALREADY IN DATASET ({summary['already_exists']}):")
        for feature in state["processed_features"]:
            if feature["normalization_status"] == "already_exists":
                print(f"   • {feature['original_feature']} → {feature['mapped_to']}")
                print(f"     Confidence: {feature['normalization_confidence']:.2f}")
    
    # Show newly discovered APIs
    if summary['new_apis_found'] > 0:
        print(f"\n🆕 NEW APIS DISCOVERED ({summary['new_apis_found']}):")
        for feature in state["processed_features"]:
            if feature["api_found"]:
                api_name = feature['api_details'].get('name', 'Unknown API')
                api_url = feature['api_details'].get('url', 'No URL')
                print(f"   • {feature['original_feature']}")
                print(f"     API: {api_name}")
                print(f"     URL: {api_url}")
                print(f"     Attempts: {feature['search_attempts']}")
    
    # Show features not found
    if summary['not_found'] > 0:
        print(f"\n❌ FEATURES NOT FOUND ({summary['not_found']}):")
        for feature in state["processed_features"]:
            if not feature["api_found"] and feature["normalization_status"] != "already_exists":
                print(f"   • {feature['original_feature']}")
                reason = feature.get('reason', 'Unknown reason')
                if len(reason) > 60:
                    reason = reason[:57] + "..."
                print(f"     Reason: {reason}")
                if feature.get('search_attempts', 0) > 0:
                    print(f"     Search attempts: {feature['search_attempts']}")
    
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS:")
    
    if summary['already_exists'] > 0:
        print(f"   • {summary['already_exists']} features are already available in your dataset")
        print("   • No additional data collection needed for these features")
    
    if summary['new_apis_found'] > 0:
        print(f"   • {summary['new_apis_found']} new APIs discovered for data collection")
        print("   • Proceed to collection agent for these features")
    
    if summary['not_found'] > 0:
        print(f"   • {summary['not_found']} features require manual investigation")
        print("   • Consider alternative data sources or feature engineering")
    
    print("="*60)


def validate_environment():
    """
    Validate required environment variables and dependencies
    """
    required_vars = ["GEMINI_API_KEY", "TAVILY_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   • {var}")
        print("\n💡 Please add these to your .env file:")
        for var in missing_vars:
            print(f"   {var}=your_api_key_here")
        return False
    
    print("✅ Environment validation passed")
    return True


def check_file_dependencies():
    """
    Check if required files exist
    """
    required_files = [
        "data/full_preprocessed_aqi_weather_data_with_all_features.csv",
        "agents/langgraph_agents/discovery_agent/prompts/feature_comparision_prompt.txt",
        "agents/langgraph_agents/discovery_agent/prompts/discovery_node_prompt.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Warning: Some files are missing (agent will attempt fallbacks):")
        for file_path in missing_files:
            print(f"   • {file_path}")
        return False
    
    print("✅ File dependencies check passed")
    return True


def main():
    """
    Main execution function with enhanced error handling and validation
    """
    print("🚀 Starting Enhanced Discovery Agent...")
    print("-" * 50)
    
    # Validate environment
    if not validate_environment():
        print("❌ Environment validation failed. Exiting.")
        return 1
    
    # Check file dependencies
    file_check = check_file_dependencies()
    if not file_check:
        print("⚠️  Some files are missing but continuing with fallbacks...")
    
    # Compile the agent
    try:
        print("\n🔧 Compiling discovery agent...")
        agent = compile_discovery_agent()
        print("✅ Agent compiled successfully")
    except Exception as e:
        print(f"❌ Failed to compile agent: {e}")
        return 1
    
    # Create initial state
    print("\n📋 Initializing agent state...")
    initial_state = create_initial_state([])  # Features will be loaded by feature_reader
    print("✅ Initial state created")
    
    # Run the agent
    try:
        print("\n🔍 Starting discovery process...")
        print("=" * 50)
        
        # Configure with higher recursion limit for complex workflows
        config = {
            "recursion_limit": 100,
            "max_execution_time": 1800  # 30 minutes timeout
        }
        
        final_state = agent.invoke(initial_state, config=config)
        
        print("\n✅ Discovery process completed successfully!")
        
        # Save results
        output_path = save_results(final_state)
        
        # Print comprehensive summary
        print_enhanced_summary(final_state)
        
        # Show next steps
        print("\n🔮 NEXT STEPS:")
        summary = get_processing_summary(final_state)
        if summary['new_apis_found'] > 0:
            print("   1. Review discovered APIs in the output file")
            print("   2. Run the collection agent for new APIs")
            print("   3. Validate collected data quality")
        if summary['already_exists'] > 0:
            print("   1. Verify feature mappings are correct")
            print("   2. Update your feature engineering pipeline")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Discovery process failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Try to save partial results
        try:
            if 'final_state' in locals():
                print("💾 Attempting to save partial results...")
                save_results(final_state)
                print("✅ Partial results saved")
        except:
            pass
        
        return 1


if __name__ == "__main__":
    # Set working directory to script location for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    exit_code = main()
    exit(exit_code)