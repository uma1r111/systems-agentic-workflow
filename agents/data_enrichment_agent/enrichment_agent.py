import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import google.generativeai as genai
from utils.api_search import APISearcher
from utils.api_fetch import APIFetcher
from utils.data_validation import DataValidator
from utils.storage import DataStorage
from utils.logger import EnrichmentLogger

class DataEnrichmentAgent:
    def __init__(self, city: str = "Karachi", max_backup_apis: int = 2):
        """
        Initialize Data Enrichment Agent
        
        Args:
            city: Target city for data collection (default: Karachi)
            max_backup_apis: Maximum number of backup APIs to try per feature
        """
        self.city = city
        self.max_backup_apis = max_backup_apis
        self.start_date = "2024-04-01"  # Fixed start date
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Initialize utility components
        self.api_searcher = APISearcher()
        self.api_fetcher = APIFetcher(timeout_seconds=15)
        self.data_validator = DataValidator()
        self.data_storage = DataStorage()
        self.logger = EnrichmentLogger()
        
        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-1.5-flash")
    
    def load_drift_results(self, drift_output_path: str) -> Dict[str, Any]:
        """Load results from Drift Monitoring Agent"""
        if not os.path.exists(drift_output_path):
            raise FileNotFoundError(f"Drift analysis results not found at {drift_output_path}")
        
        with open(drift_output_path, "r") as f:
            return json.load(f)
    
    def discover_apis_for_feature(self, feature_name: str) -> Dict[str, Any]:
        """Use LLM + Tavily to discover APIs for a specific feature"""
        prompt_template_path = Path("agents/data_enrichment_agent/prompts/api_discovery_prompt.txt")
        
        if not prompt_template_path.exists():
            raise FileNotFoundError(f"Prompt template not found at {prompt_template_path}")
        
        with open(prompt_template_path, "r") as f:
            prompt_template = f.read()
        
        # Fill in the prompt template
        prompt = prompt_template.replace("{feature_name}", feature_name)\
                               .replace("{city}", self.city)\
                               .replace("{date_range}", f"{self.start_date} to {self.end_date}")
        
        # First, use Tavily to search for APIs
        search_results = self.api_searcher.search_apis(feature_name, self.city)
        
        # Combine prompt with search results for LLM analysis
        full_prompt = f"{prompt}\n\nTavily Search Results:\n{json.dumps(search_results, indent=2)}"
        
        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=800,
                    response_mime_type="application/json"
                )
            )
            
            api_recommendations = json.loads(response.text)
            return api_recommendations
            
        except Exception as e:
            self.logger.log_error(f"LLM API discovery failed for {feature_name}: {str(e)}")
            return {
                "primary_api": None,
                "backup_apis": [],
                "documentation_links": [],
                "confidence": "low",
                "reasoning": f"API discovery failed: {str(e)}"
            }
    
    def fetch_feature_data(self, feature_name: str, api_config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Fetch data for a single feature using primary + backup APIs"""
        apis_to_try = []
        
        # Add primary API
        if api_config.get("primary_api"):
            apis_to_try.append(api_config["primary_api"])
        
        # Add backup APIs
        backup_apis = api_config.get("backup_apis", [])[:self.max_backup_apis]
        apis_to_try.extend(backup_apis)
        
        for i, api_info in enumerate(apis_to_try):
            api_type = "primary" if i == 0 else f"backup_{i}"
            
            try:
                self.logger.log_info(f"Attempting {api_type} API for {feature_name}: {api_info.get('name', 'Unknown')}")
                
                # Fetch data using the API
                raw_data = self.api_fetcher.fetch_data(
                    api_info, 
                    feature_name, 
                    self.city, 
                    self.start_date, 
                    self.end_date
                )
                
                if raw_data is not None and len(raw_data) > 0:
                    # Validate the data
                    validated_data = self.data_validator.validate_and_clean(raw_data, feature_name)
                    
                    if validated_data is not None:
                        self.logger.log_success(
                            feature_name=feature_name,
                            api_used=api_info.get('name', 'Unknown'),
                            api_type=api_type,
                            record_count=len(validated_data)
                        )
                        return validated_data
                    else:
                        self.logger.log_warning(f"Data validation failed for {feature_name} using {api_type}")
                else:
                    self.logger.log_warning(f"No data returned from {api_type} for {feature_name}")
                    
            except Exception as e:
                self.logger.log_error(f"{api_type} API failed for {feature_name}: {str(e)}")
                continue
        
        # All APIs failed
        self.logger.log_failure(
            feature_name=feature_name,
            reason="All APIs failed or returned invalid data"
        )
        return None
    
    def process_single_feature(self, feature_name: str) -> Dict[str, Any]:
        """Process a single feature: discover APIs, fetch data, validate, and store"""
        self.logger.log_info(f"Processing feature: {feature_name}")
        
        # Step 1: Discover APIs
        api_config = self.discover_apis_for_feature(feature_name)
        
        if not api_config.get("primary_api"):
            return {
                "feature_name": feature_name,
                "status": "failed",
                "reason": "No suitable APIs found",
                "data_file": None
            }
        
        # Step 2: Fetch and validate data
        feature_data = self.fetch_feature_data(feature_name, api_config)
        
        if feature_data is None:
            return {
                "feature_name": feature_name,
                "status": "failed",
                "reason": "Data fetch failed for all APIs",
                "data_file": None
            }
        
        # Step 3: Store data
        try:
            output_file = self.data_storage.save_feature_data(
                feature_data, 
                feature_name, 
                self.city
            )
            
            return {
                "feature_name": feature_name,
                "status": "success",
                "reason": f"Successfully fetched {len(feature_data)} records",
                "data_file": output_file,
                "record_count": len(feature_data)
            }
            
        except Exception as e:
            self.logger.log_error(f"Failed to save data for {feature_name}: {str(e)}")
            return {
                "feature_name": feature_name,
                "status": "failed",
                "reason": f"Data storage failed: {str(e)}",
                "data_file": None
            }
    
    def run_enrichment(self, drift_output_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Main orchestration method: process all features from drift analysis
        
        Args:
            drift_output_path: Path to drift monitoring agent output JSON
            output_path: Path to save enrichment results
        
        Returns:
            Dictionary with enrichment results and summary
        """
        self.logger.log_info("Starting data enrichment process")
        
        # Load drift analysis results
        try:
            drift_results = self.load_drift_results(drift_output_path)
            features_to_add = drift_results.get("features_to_add", [])
            
            if not features_to_add:
                self.logger.log_info("No features to add - drift analysis found no significant changes")
                return {
                    "enrichment_triggered": False,
                    "reason": "No features recommended by drift analysis",
                    "processed_features": [],
                    "summary": {"total": 0, "success": 0, "failed": 0}
                }
            
            self.logger.log_info(f"Processing {len(features_to_add)} features: {features_to_add}")
            
        except Exception as e:
            self.logger.log_error(f"Failed to load drift results: {str(e)}")
            return {
                "enrichment_triggered": False,
                "reason": f"Failed to load drift results: {str(e)}",
                "processed_features": [],
                "summary": {"total": 0, "success": 0, "failed": 0}
            }
        
        # Process each feature
        processed_features = []
        success_count = 0
        
        for feature_name in features_to_add:
            try:
                result = self.process_single_feature(feature_name)
                processed_features.append(result)
                
                if result["status"] == "success":
                    success_count += 1
                    
            except Exception as e:
                self.logger.log_error(f"Unexpected error processing {feature_name}: {str(e)}")
                processed_features.append({
                    "feature_name": feature_name,
                    "status": "failed",
                    "reason": f"Unexpected error: {str(e)}",
                    "data_file": None
                })
        
        # Create summary
        summary = {
            "total": len(features_to_add),
            "success": success_count,
            "failed": len(features_to_add) - success_count,
            "success_rate": f"{(success_count/len(features_to_add)*100):.1f}%" if features_to_add else "0%"
        }
        
        # Compile final output
        output = {
            "enrichment_triggered": True,
            "city": self.city,
            "date_range": f"{self.start_date} to {self.end_date}",
            "processed_features": processed_features,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save output if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2, default=str)
        
        self.logger.log_info(f"Enrichment complete. Success rate: {summary['success_rate']}")
        return output


if __name__ == "__main__":
    # Example usage
    agent = DataEnrichmentAgent(city="Karachi", max_backup_apis=2)
    
    result = agent.run_enrichment(
        drift_output_path="agents/drift_monitoring_agent/outputs/drift_analysis.json",
        output_path="agents/data_enrichment_agent/outputs/enrichment_results.json"
    )
    
    print("Data Enrichment Complete!")
    print(f"Enrichment triggered: {result['enrichment_triggered']}")
    print(f"Summary: {result.get('summary', {})}")
    
    # Print feature-by-feature results
    for feature_result in result.get('processed_features', []):
        print(f"  {feature_result['feature_name']}: {feature_result['status']} - {feature_result['reason']}")