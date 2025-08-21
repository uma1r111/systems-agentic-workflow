"""
Discovery Agent Nodes for LangGraph
Contains the core functionality nodes for API discovery workflow with feature normalization
"""

import os
import json
import csv
import requests
import google.generativeai as genai
from typing import Dict, Any, List
from tavily import TavilyClient
from pathlib import Path
from discovery_state import (
    DiscoveryState, 
    get_next_feature, 
    update_feature_processing, 
    increment_search_attempt,
    update_normalization_result,
    load_csv_columns
)


def find_file_in_paths(filename: str, search_paths: List[str]) -> str:
    """
    Search for a file in multiple possible paths
    """
    for path in search_paths:
        full_path = Path(path)
        if full_path.exists():
            print(f"✓ Found file at: {full_path}")
            return str(full_path)
    
    print(f"✗ File '{filename}' not found in any of these paths:")
    for path in search_paths:
        print(f"   • {path}")
    return None


def get_project_root() -> Path:
    """
    Get the project root directory by looking for common project markers
    """
    current = Path.cwd()
    
    # Look for common project markers
    markers = ['.git', 'pyproject.toml', 'setup.py', 'requirements.txt', '.env']
    
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    
    # Fallback to current directory
    return current


def feature_reader(state: DiscoveryState) -> DiscoveryState:
    """
    Read next feature from drift analysis or get next feature to process
    """
    print(f"📋 Feature reader called. Current index: {state['current_feature_index']}, Total: {state['total_features']}")
    
    if state["current_feature_index"] == 0 and state["total_features"] == 0:
        # First run - read from drift analysis file
        try:
            # Get project root and current working directory
            project_root = get_project_root()
            current_dir = Path.cwd()
            
            print(f"🔍 Project root: {project_root}")
            print(f"🔍 Current directory: {current_dir}")
            
            # Multiple possible paths for drift analysis file
            possible_paths = [
                # From current directory
                "agents/drift_monitoring_agent/outputs/drift_analysis.json",
                "drift_monitoring_agent/outputs/drift_analysis.json",
                "outputs/drift_analysis.json",
                "drift_analysis.json",
                
                # From project root
                project_root / "agents" / "drift_monitoring_agent" / "outputs" / "drift_analysis.json",
                project_root / "drift_monitoring_agent" / "outputs" / "drift_analysis.json",
                project_root / "outputs" / "drift_analysis.json",
                project_root / "drift_analysis.json",
                
                # Relative to script location
                Path(__file__).parent / "outputs" / "drift_analysis.json",
                Path(__file__).parent.parent / "drift_monitoring_agent" / "outputs" / "drift_analysis.json",
                Path(__file__).parent.parent.parent / "drift_monitoring_agent" / "outputs" / "drift_analysis.json",
            ]
            
            # Convert all paths to strings for consistency
            possible_paths = [str(path) for path in possible_paths]
            
            drift_analysis_path = find_file_in_paths("drift_analysis.json", possible_paths)
            
            if not drift_analysis_path:
                # Create a sample drift analysis for testing
                print("⚠️  Creating sample drift analysis for testing...")
                sample_drift = {
                    "features_to_add": [
                        "air_quality_index",
                        "pm2_5_concentration", 
                        "ozone_levels",
                        "nitrogen_dioxide",
                        "sulfur_dioxide"
                    ],
                    "timestamp": "2024-01-01T00:00:00",
                    "analysis_type": "sample_data"
                }
                
                # Save sample to outputs directory
                output_dir = Path("outputs")
                output_dir.mkdir(exist_ok=True)
                sample_path = output_dir / "drift_analysis.json"
                
                with open(sample_path, "w") as f:
                    json.dump(sample_drift, f, indent=2)
                
                drift_analysis_path = str(sample_path)
                print(f"✓ Created sample drift analysis at: {drift_analysis_path}")
            
            # Load the drift analysis file
            with open(drift_analysis_path, "r") as f:
                drift_data = json.load(f)
            
            features_to_add = drift_data.get("features_to_add", [])
            state["features_to_process"] = features_to_add
            state["total_features"] = len(features_to_add)
            
            print(f"✓ Loaded {len(features_to_add)} features to process: {features_to_add}")
            
        except Exception as e:
            state["status"] = "error"
            state["reason"] = f"Failed to read drift analysis: {e}"
            print(f"✗ Error loading drift analysis: {e}")
            return state
    
    # Get next feature to process
    next_feature = get_next_feature(state)
    print(f"📋 Next feature to process: {next_feature}")
    
    if next_feature:
        state = update_feature_processing(state, next_feature)
        state["status"] = "normalizing"
        print(f"🔄 Processing feature: {next_feature}")
    else:
        state["status"] = "completed"
        print("✅ All features processed - ending workflow")
    
    return state


def feature_normalizer(state: DiscoveryState) -> DiscoveryState:
    """
    Check if the requested feature already exists in the CSV under a different name
    Uses LLM for semantic comparison of feature names
    """
    feature = state["feature"]
    print(f"🔍 Feature normalizer called for: {feature}")
    
    # Load CSV columns if not already loaded
    if not state["csv_columns_loaded"]:
        try:
            # Get project root for better path resolution
            project_root = get_project_root()
            
            # Multiple possible paths for the CSV file
            csv_paths = [
                "data/full_preprocessed_aqi_weather_data_with_all_features.csv",
                "../data/full_preprocessed_aqi_weather_data_with_all_features.csv",
                project_root / "data" / "full_preprocessed_aqi_weather_data_with_all_features.csv",
                "full_preprocessed_aqi_weather_data_with_all_features.csv",
                "sample_data.csv"  # Fallback
            ]
            
            csv_paths = [str(path) for path in csv_paths]
            csv_path = find_file_in_paths("CSV data file", csv_paths)
            
            if not csv_path:
                # Create sample CSV columns for testing
                print("⚠️  Creating sample CSV columns for testing...")
                sample_columns = [
                    "timestamp", "temperature", "humidity", "pressure", "wind_speed", "wind_direction",
                    "precipitation", "visibility", "uv_index", "dew_point", "feels_like_temperature",
                    "cloud_cover", "solar_radiation", "air_pressure_sea_level", "pm10", "co2_levels"
                ]
                state = load_csv_columns(state, sample_columns)
                print(f"✓ Loaded {len(sample_columns)} sample columns")
            else:
                # Read actual CSV columns
                with open(csv_path, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    columns = next(csv_reader)  # Get header row
                
                state = load_csv_columns(state, columns)
                print(f"✓ Loaded {len(columns)} columns from CSV: {csv_path}")
                print(f"Sample columns: {columns[:10]}")
            
        except Exception as e:
            state = update_normalization_result(
                state, "error",
                reason=f"Failed to load CSV columns: {e}"
            )
            print(f"✗ Error loading CSV: {e}")
            return state
    
    # Configure Gemini for semantic comparison
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        state = update_normalization_result(
            state, "error",
            reason=f"Failed to configure Gemini: {e}"
        )
        print(f"✗ Gemini configuration error: {e}")
        return state
    
    # Load feature comparison prompt from file or use default
    project_root = get_project_root()
    prompt_paths = [
        "agents/langgraph_agents/discovery_agent/prompts/feature_comparision_prompt.txt",
        "prompts/feature_comparision_prompt.txt",
        project_root / "agents" / "langgraph_agents" / "discovery_agent" / "prompts" / "feature_comparision_prompt.txt",
        Path(__file__).parent / "prompts" / "feature_comparision_prompt.txt",
    ]
    
    prompt_paths = [str(path) for path in prompt_paths]
    prompt_file = find_file_in_paths("feature comparison prompt", prompt_paths)
    
    if prompt_file:
        with open(prompt_file, "r", encoding='utf-8') as f:
            prompt_template = f.read()
        print(f"✓ Loaded feature comparison prompt from: {prompt_file}")
    else:
        # Use default prompt template
        prompt_template = """
        You are analyzing whether a requested feature already exists in a dataset under a different name.

        Requested Feature: {feature}
        Existing Columns: {existing_columns}

        Please analyze if the requested feature matches any existing column name semantically, even if the names are different.

        Return a JSON response with:
        {{
            "exists": true/false,
            "confidence": 0.0-1.0,
            "matched_column": "column_name" (if exists=true),
            "similarity_scores": {{"column1": score, "column2": score}},
            "reasoning": "explanation of your analysis"
        }}

        Use confidence > 0.7 for strong matches only.
        """
        print("⚠️  Using default feature comparison prompt template")
    
    # Format the prompt with feature and existing columns
    existing_columns = state["existing_columns"]
    
    try:
        normalization_prompt = prompt_template.format(
            feature=feature,
            existing_columns=', '.join(existing_columns)
        )
        print(f"✓ Feature comparison prompt formatted for: {feature}")
    except Exception as e:
        state = update_normalization_result(
            state, "error",
            reason=f"Failed to format feature comparison prompt: {e}"
        )
        print(f"✗ Feature comparison prompt formatting error: {e}")
        return state
    
    try:
        print("🤖 Calling Gemini for feature normalization...")
        response = model.generate_content(normalization_prompt)
        response_text = response.text.strip()
        
        print(f"✓ Gemini response received")
        print(f"Raw response (first 200 chars): {response_text[:200]}")
        
        # Extract JSON from response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "{" in response_text:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            response_text = response_text[json_start:json_end]
        
        result = json.loads(response_text)
        
        # Process normalization result
        if result.get("exists", False) and result.get("confidence", 0) > 0.7:
            state = update_normalization_result(
                state,
                normalization_status="already_exists",
                mapped_to=result.get("matched_column"),
                similarity_scores=result.get("similarity_scores", {}),
                confidence=result.get("confidence", 0.0),
                reason=result.get("reasoning", "Feature already exists in dataset")
            )
            print(f"✅ Feature already exists: {feature} → {result.get('matched_column')}")
        else:
            state = update_normalization_result(
                state,
                normalization_status="missing",
                similarity_scores=result.get("similarity_scores", {}),
                confidence=result.get("confidence", 0.0),
                reason=result.get("reasoning", "Feature not found in existing columns")
            )
            print(f"🔍 Feature is missing, proceeding to API search: {feature}")
            
    except json.JSONDecodeError as e:
        state = update_normalization_result(
            state, "error",
            reason=f"Invalid JSON response from normalization LLM: {e}"
        )
        print(f"✗ JSON parsing error in normalization: {e}")
    except Exception as e:
        state = update_normalization_result(
            state, "error",
            reason=f"Normalization failed: {str(e)}"
        )
        print(f"✗ Normalization error: {e}")
    
    return state


def api_searcher(state: DiscoveryState) -> DiscoveryState:
    """
    Search for APIs using Gemini LLM combined with Tavily web search
    """
    feature = state["feature"]
    state = increment_search_attempt(state)
    
    print(f"🔍 API search attempt {state['search_attempts']} for: {feature}")
    
    # Configure Tavily for web search
    try:
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        print("✓ Tavily client configured")
    except Exception as e:
        state["status"] = "error"
        state["reason"] = f"Failed to configure Tavily: {e}"
        print(f"✗ Tavily configuration error: {e}")
        return state
    
    # Configure Gemini
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        print("✓ Gemini configured")
    except Exception as e:
        state["status"] = "error"
        state["reason"] = f"Failed to configure Gemini: {e}"
        print(f"✗ Gemini configuration error: {e}")
        return state
    
    # Search for APIs using Tavily
    try:
        # Create search query for the feature
        search_queries = [
            f"{feature} API weather environmental data",
            f"{feature} open data API free",
            f"environmental {feature} API Pakistan Karachi"
        ]
        
        web_results = []
        for query in search_queries:
            try:
                print(f"🌐 Searching web with Tavily: {query}")
                search_result = tavily_client.search(
                    query=query,
                    search_depth="basic",
                    max_results=5,
                    include_domains=["api.openweathermap.org", "api.weather.gov", "opendata.gov", "data.gov", "api.worldbank.org"]
                )
                
                if search_result.get("results"):
                    web_results.extend(search_result["results"])
                    print(f"✓ Found {len(search_result['results'])} results for query")
                
            except Exception as e:
                print(f"⚠️ Tavily search failed for query '{query}': {e}")
                continue
        
        if not web_results:
            print("⚠️ No web results found, falling back to LLM-only search")
        else:
            print(f"✓ Total web results collected: {len(web_results)}")
        
    except Exception as e:
        print(f"⚠️ Tavily search failed completely: {e}")
        web_results = []
    
    # Load prompt from file or use default
    project_root = get_project_root()
    prompt_paths = [
        "agents/langgraph_agents/discovery_agent/prompts/discovery_node_prompt.txt",
        "prompts/discovery_node_prompt.txt",
        project_root / "agents" / "langgraph_agents" / "discovery_agent" / "prompts" / "discovery_node_prompt.txt",
        Path(__file__).parent / "prompts" / "discovery_node_prompt.txt",
    ]
    
    prompt_paths = [str(path) for path in prompt_paths]
    prompt_file = find_file_in_paths("discovery prompt", prompt_paths)
    
    if prompt_file:
        with open(prompt_file, "r", encoding='utf-8') as f:
            prompt_template = f.read()
        print(f"✓ Loaded prompt from: {prompt_file}")
    else:
        # Use default prompt template
        prompt_template = """
        You are an API discovery agent. Find APIs that can provide data for the requested feature.

        Requested Feature: {feature}
        Web Search Results: {web_results}

        Please search for and recommend APIs that can provide data for this feature.
        Focus on free, reliable, and well-documented APIs.

        Return a JSON response with:
        {{
            "search_keywords": ["keyword1", "keyword2"],
            "candidate_api": {{
                "name": "API Name",
                "url": "https://api.example.com",
                "endpoint": "/data/endpoint",
                "description": "API description",
                "data_format": "JSON/XML",
                "authentication": "API Key/OAuth/None",
                "rate_limits": "rate limit info",
                "documentation": "docs URL"
            }},
            "confidence": 0.0-1.0,
            "reason": "explanation of choice or why none found"
        }}

        If no suitable API is found, set candidate_api to null and explain why.
        """
        print("⚠️  Using default discovery prompt template")
    
    # Format web results for prompt
    web_results_text = ""
    if web_results:
        web_results_text = "\n".join([
            f"- {result.get('title', 'No title')}: {result.get('url', '')} - {result.get('content', '')[:200]}..."
            for result in web_results[:10]  # Limit to top 10 results
        ])
    else:
        web_results_text = "No specific web results found. Please suggest based on your knowledge."
    
    # Format the prompt
    try:
        prompt = prompt_template.format(
            feature=feature,
            web_results=web_results_text
        )
        print(f"✓ Prompt formatted for feature: {feature}")
    except Exception as e:
        state["status"] = "error"
        state["reason"] = f"Failed to format prompt: {e}"
        print(f"✗ Prompt formatting error: {e}")
        return state
    
    # Call Gemini API with web search context
    try:
        print("🤖 Calling Gemini API with web search context...")
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        print(f"✓ Gemini response received for {feature}")
        print(f"Raw response (first 300 chars): {response_text[:300]}")
        
        # Try to extract JSON from response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "{" in response_text:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            response_text = response_text[json_start:json_end]
        
        result = json.loads(response_text)
        
        state["search_keywords"] = result.get("search_keywords", [])
        state["candidate_api"] = result.get("candidate_api")
        state["confidence_score"] = result.get("confidence", 0.0)
        
        if state["candidate_api"]:
            state["status"] = "validating"
            print(f"✓ Found candidate API: {state['candidate_api'].get('name', 'Unknown')}")
            print(f"  URL: {state['candidate_api'].get('url', 'No URL')}")
        else:
            state["status"] = "not_found"
            state["result_status"] = "not_found"
            state["reason"] = result.get("reason", "No suitable API found")
            print(f"✗ No API found for {feature}")
            
    except json.JSONDecodeError as e:
        state["status"] = "error"
        state["reason"] = f"Invalid JSON response from LLM: {e}"
        print(f"✗ JSON parsing error: {e}")
        if 'response_text' in locals():
            print(f"Response text: {response_text[:500]}")
    except Exception as e:
        state["status"] = "error"
        state["reason"] = f"Search failed: {str(e)}"
        print(f"✗ Search error: {e}")
    
    return state


def api_validator(state: DiscoveryState) -> DiscoveryState:
    """
    Validate the candidate API
    """
    candidate_api = state["candidate_api"]
    
    print(f"🔍 API validator called for: {candidate_api.get('name', 'Unknown') if candidate_api else 'None'}")
    
    if not candidate_api:
        state["status"] = "invalid"
        state["reason"] = "No candidate API to validate"
        print("✗ No candidate API to validate")
        return state
    
    try:
        # Basic URL validation
        url = candidate_api.get("url", "")
        if not url.startswith(("http://", "https://")):
            state["status"] = "invalid"
            state["reason"] = "Invalid URL format"
            print(f"✗ Invalid URL format: {url}")
            return state
        
        # Test API accessibility
        test_url = url
        endpoint = candidate_api.get("endpoint", "")
        if endpoint:
            test_url = f"{url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        print(f"🌐 Testing API accessibility: {test_url}")
        
        # Try a basic request (with timeout)
        headers = {
            'User-Agent': 'DiscoveryAgent/1.0'
        }
        response = requests.get(test_url, timeout=15, headers=headers)
        
        print(f"📡 API response status: {response.status_code}")
        
        if response.status_code == 200:
            state["validated_api"] = candidate_api
            state["status"] = "valid"
            state["result_status"] = "found"
            print(f"✅ API validated successfully: {candidate_api.get('name')}")
        elif response.status_code == 401:
            # API exists but needs auth - still valid
            state["validated_api"] = candidate_api
            state["status"] = "valid"
            state["result_status"] = "found"
            state["reason"] = "API requires authentication (valid but needs API key)"
            print(f"🔑 API requires authentication but is valid: {candidate_api.get('name')}")
        elif response.status_code == 403:
            # Forbidden - API exists but access denied
            state["validated_api"] = candidate_api
            state["status"] = "valid"
            state["result_status"] = "found"
            state["reason"] = "API exists but access forbidden (may need API key or different endpoint)"
            print(f"🚫 API access forbidden but exists: {candidate_api.get('name')}")
        elif response.status_code in [404, 410]:
            state["status"] = "invalid"
            state["reason"] = f"API endpoint not found (status: {response.status_code})"
            print(f"✗ API endpoint not found: {response.status_code}")
        else:
            state["status"] = "invalid"
            state["reason"] = f"API returned unexpected status: {response.status_code}"
            print(f"⚠️ Unexpected API status: {response.status_code}")
            
    except requests.exceptions.Timeout:
        state["status"] = "invalid"
        state["reason"] = "API timeout - server not responding within 15 seconds"
        print("⏰ API timeout")
    except requests.exceptions.ConnectionError:
        state["status"] = "invalid"
        state["reason"] = "API connection failed - server unreachable"
        print("🔌 API connection failed")
    except requests.exceptions.RequestException as e:
        state["status"] = "invalid"
        state["reason"] = f"API request failed: {str(e)}"
        print(f"📡 API request failed: {e}")
    except Exception as e:
        state["status"] = "invalid"
        state["reason"] = f"Validation error: {str(e)}"
        print(f"✗ Validation error: {e}")
    
    return state