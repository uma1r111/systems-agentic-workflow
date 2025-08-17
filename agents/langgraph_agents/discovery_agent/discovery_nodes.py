"""
Discovery Agent Nodes for LangGraph
Contains the core functionality nodes for API discovery workflow
"""

import os
import json
import requests
import google.generativeai as genai
from typing import Dict, Any
from discovery_state import DiscoveryState, get_next_feature, update_feature_processing, increment_search_attempt


def feature_reader(state: DiscoveryState) -> DiscoveryState:
    """
    Read next feature from drift analysis or get next feature to process
    """
    print(f" Feature reader called. Current index: {state['current_feature_index']}, Total: {state['total_features']}")
    
    if state["current_feature_index"] == 0 and state["total_features"] == 0:
        # First run - read from drift analysis file
        try:
            # Try multiple possible paths
            possible_paths = [
        
                "agents/drift_monitoring_agent/outputs/drift_analysis.json"
                
            ]
            
            drift_data = None
            for path in possible_paths:
                try:
                    with open(path, "r") as f:
                        drift_data = json.load(f)
                    print(f"✓ Found drift analysis at: {path}")
                    break
                except FileNotFoundError:
                    continue
            
            if not drift_data:
                raise FileNotFoundError("Drift analysis file not found in any expected location")
            
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
        state["status"] = "searching"
        print(f"🔍 Processing feature: {next_feature}")
    else:
        state["status"] = "completed"
        print("✅ All features processed - ending workflow")
    
    return state


def api_searcher(state: DiscoveryState) -> DiscoveryState:
    """
    Search for APIs using Gemini LLM and web search
    """
    feature = state["feature"]
    state = increment_search_attempt(state)
    
    print(f"🔍 API search attempt {state['search_attempts']} for: {feature}")
    
    # Configure Gemini
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        state["status"] = "error"
        state["reason"] = f"Failed to configure Gemini: {e}"
        print(f"✗ Gemini configuration error: {e}")
        return state
    
    # Load prompt from file
    prompt_template = None
    prompt_paths = [
        "agents/langgraph_agents/discovery_agent/prompts/discovery_node_prompt.txt"
    ]
    
    for path in prompt_paths:
        try:
            with open(path, "r", encoding='utf-8') as f:
                prompt_template = f.read()
            print(f"✓ Loaded prompt from: {path}")
            break
        except FileNotFoundError:
            continue
    
    if not prompt_template:
        state["status"] = "error"
        state["reason"] = "Prompt template not found in any location"
        print("✗ Prompt template not found")
        return state
    
    # Format the prompt
    try:
        prompt = prompt_template.format(feature=feature)
        print(f"✓ Prompt formatted for feature: {feature}")
        print(f"Prompt length: {len(prompt)} characters")
    except Exception as e:
        state["status"] = "error"
        state["reason"] = f"Failed to format prompt: {e}"
        print(f"✗ Prompt formatting error: {e}")
        return state
    
    # Call Gemini API
    try:
        print("📡 Calling Gemini API...")
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        print(f"✓ Gemini response received for {feature}")
        print(f"Raw response (first 300 chars): {response_text[:300]}")
        
        # Try to extract JSON from response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
            print("✓ Extracted JSON from markdown")
        elif "{" in response_text:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            response_text = response_text[json_start:json_end]
            print("✓ Extracted JSON from text")
        
        print(f"JSON to parse: {response_text[:200]}...")
        
        result = json.loads(response_text)
        
        state["search_keywords"] = result.get("search_keywords", [])
        state["candidate_api"] = result.get("candidate_api")
        state["confidence_score"] = result.get("confidence", 0.0)
        
        if state["candidate_api"]:
            state["status"] = "validating"
            print(f"✓ Found candidate API: {state['candidate_api'].get('name', 'Unknown')}")
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
            print(f"Full response text: {response_text}")
    except Exception as e:
        state["status"] = "error"
        state["reason"] = f"Search failed: {str(e)}"
        print(f"✗ Search error: {e}")
        if 'response' in locals():
            print(f"Full response: {response.text}")
    
    return state


def api_validator(state: DiscoveryState) -> DiscoveryState:
    """
    Validate the candidate API
    """
    candidate_api = state["candidate_api"]
    
    if not candidate_api:
        state["status"] = "invalid"
        state["reason"] = "No candidate API to validate"
        return state
    
    try:
        # Basic URL validation
        url = candidate_api.get("url", "")
        if not url.startswith(("http://", "https://")):
            state["status"] = "invalid"
            state["reason"] = "Invalid URL format"
            return state
        
        # Test API accessibility
        test_url = url
        if candidate_api.get("endpoint"):
            test_url = f"{url.rstrip('/')}/{candidate_api['endpoint'].lstrip('/')}"
        
        # Try a basic request (with timeout)
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            state["validated_api"] = candidate_api
            state["status"] = "valid"
            state["result_status"] = "found"
        elif response.status_code == 401:
            # API exists but needs auth - still valid for free tier
            state["validated_api"] = candidate_api
            state["status"] = "valid"
            state["result_status"] = "found"
            state["reason"] = "API requires authentication"
        else:
            state["status"] = "invalid"
            state["reason"] = f"API returned status code: {response.status_code}"
            
    except requests.exceptions.Timeout:
        state["status"] = "invalid"
        state["reason"] = "API timeout - server not responding"
    except requests.exceptions.RequestException as e:
        state["status"] = "invalid"
        state["reason"] = f"API connection failed: {str(e)}"
    except Exception as e:
        state["status"] = "invalid"
        state["reason"] = f"Validation error: {str(e)}"
    
    return state