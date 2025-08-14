import os
import json
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime
import time

class APISearcher:
    def __init__(self):
        """Initialize API Searcher with Tavily API"""
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")
        
        self.tavily_base_url = "https://api.tavily.com/search"
        self.max_results = 10
        self.search_depth = "basic"
        
    def _construct_search_query(self, feature_name: str, city: str) -> str:
        """Construct optimized search query for finding APIs"""
        # Create feature-specific search terms
        feature_keywords = {
            # Weather features
            "temperature": "weather temperature API",
            "humidity": "weather humidity API", 
            "wind_speed": "wind speed weather API",
            "precipitation": "rainfall precipitation API",
            "pressure": "atmospheric pressure API",
            
            # Air quality features
            "pm2_5": "PM2.5 air quality API",
            "pm10": "PM10 air quality API",
            "no2": "nitrogen dioxide air quality API",
            "so2": "sulfur dioxide air quality API", 
            "co": "carbon monoxide air quality API",
            "o3": "ozone air quality API",
            "aqi": "air quality index API",
            
            # Derived features
            "visibility": "visibility weather API",
            "uv_index": "UV index weather API",
            "pollen": "pollen count API",
            "traffic": "traffic data API"
        }
        
        # Get specific keywords or use generic approach
        base_keywords = feature_keywords.get(feature_name.lower(), f"{feature_name} API")
        
        # Construct comprehensive search query
        search_terms = [
            f"{base_keywords} {city} Pakistan",
            f"{base_keywords} historical data",
            f"{base_keywords} JSON REST",
            f"free {base_keywords}",
            f"public {base_keywords}"
        ]
        
        return " OR ".join(search_terms)
    
    def search_apis(self, feature_name: str, city: str) -> Dict[str, Any]:
        """
        Search for APIs using Tavily that can provide the specified feature data
        
        Args:
            feature_name: Name of the feature to find APIs for
            city: Target city for data collection
            
        Returns:
            Dictionary with search results and metadata
        """
        query = self._construct_search_query(feature_name, city)
        
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": self.search_depth,
            "include_answer": True,
            "include_raw_content": False,
            "max_results": self.max_results,
            "include_domains": [
                "api.openweathermap.org",
                "api.weatherapi.com", 
                "api.airvisual.com",
                "api.waqi.info",
                "rapidapi.com",
                "docs.github.io",
                "opendata.gov.pk",
                "pmd.gov.pk"
            ]
        }
        
        try:
            response = requests.post(
                self.tavily_base_url,
                headers={
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                search_results = response.json()
                return self._process_search_results(search_results, feature_name, city)
            else:
                return {
                    "status": "error",
                    "error": f"Tavily API returned status {response.status_code}",
                    "results": [],
                    "query_used": query
                }
                
        except requests.RequestException as e:
            return {
                "status": "error", 
                "error": f"Tavily API request failed: {str(e)}",
                "results": [],
                "query_used": query
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Unexpected error during API search: {str(e)}",
                "results": [],
                "query_used": query
            }
    
    def _process_search_results(self, raw_results: Dict[str, Any], feature_name: str, city: str) -> Dict[str, Any]:
        """Process and filter Tavily search results for API relevance"""
        processed_results = []
        
        results = raw_results.get("results", [])
        
        for result in results:
            processed_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),
                "api_indicators": self._detect_api_indicators(result),
                "geographic_relevance": self._assess_geographic_relevance(result, city),
                "feature_relevance": self._assess_feature_relevance(result, feature_name)
            }
            
            # Only include results that show API indicators
            if processed_result["api_indicators"]["is_likely_api"]:
                processed_results.append(processed_result)
        
        # Sort by relevance score
        processed_results.sort(
            key=lambda x: (
                x["api_indicators"]["confidence_score"] * 0.4 +
                x["geographic_relevance"] * 0.3 + 
                x["feature_relevance"] * 0.3
            ), 
            reverse=True
        )
        
        return {
            "status": "success",
            "query_used": raw_results.get("query", ""),
            "total_results": len(results),
            "api_candidates": processed_results[:8],  # Top 8 most relevant
            "search_timestamp": datetime.now().isoformat(),
            "tavily_answer": raw_results.get("answer", "")
        }
    
    def _detect_api_indicators(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if a search result likely represents an API"""
        title = result.get("title", "").lower()
        url = result.get("url", "").lower() 
        content = result.get("content", "").lower()
        
        api_keywords = [
            "api", "rest", "json", "endpoint", "documentation", "docs",
            "developer", "integration", "sdk", "curl", "request", "response"
        ]
        
        data_keywords = [
            "weather", "climate", "air quality", "pollution", "environmental",
            "meteorological", "atmospheric", "historical data", "real-time"
        ]
        
        negative_keywords = [
            "news", "article", "blog", "forum", "discussion", "tutorial",
            "deprecated", "discontinued", "offline"
        ]
        
        # Calculate scores
        api_score = sum(1 for keyword in api_keywords if keyword in (title + content))
        data_score = sum(1 for keyword in data_keywords if keyword in (title + content))
        negative_score = sum(1 for keyword in negative_keywords if keyword in (title + content))
        
        # URL-based indicators
        url_indicators = any(indicator in url for indicator in ["api.", "/api/", "docs.", "/docs/", "developer"])
        
        confidence_score = (api_score + data_score + (2 if url_indicators else 0) - negative_score) / 10
        confidence_score = max(0, min(1, confidence_score))  # Clamp between 0 and 1
        
        return {
            "is_likely_api": confidence_score > 0.3,
            "confidence_score": confidence_score,
            "api_keywords_found": api_score,
            "data_keywords_found": data_score,
            "url_indicators": url_indicators,
            "negative_indicators": negative_score
        }
    
    def _assess_geographic_relevance(self, result: Dict[str, Any], city: str) -> float:
        """Assess how relevant the result is for the target city"""
        content = (result.get("title", "") + " " + result.get("content", "")).lower()
        
        # Geographic keywords (weighted by relevance)
        geo_keywords = {
            city.lower(): 1.0,
            "pakistan": 0.8,
            "karachi": 1.0 if city.lower() == "karachi" else 0.5,
            "south asia": 0.6,
            "asia": 0.4,
            "global": 0.7,
            "worldwide": 0.7,
            "international": 0.6
        }
        
        relevance_score = 0.0
        for keyword, weight in geo_keywords.items():
            if keyword in content:
                relevance_score = max(relevance_score, weight)
        
        return relevance_score
    
    def _assess_feature_relevance(self, result: Dict[str, Any], feature_name: str) -> float:
        """Assess how relevant the result is for the target feature"""
        content = (result.get("title", "") + " " + result.get("content", "")).lower()
        feature_lower = feature_name.lower()
        
        # Direct feature match
        if feature_lower in content:
            return 1.0
        
        # Feature category matching
        weather_features = ["temperature", "humidity", "wind", "precipitation", "pressure"]
        air_quality_features = ["pm2.5", "pm10", "no2", "so2", "co", "o3", "aqi"]
        
        if any(f in feature_lower for f in weather_features):
            if "weather" in content or "meteorological" in content:
                return 0.8
        
        if any(f in feature_lower for f in air_quality_features):
            if "air quality" in content or "pollution" in content or "environmental" in content:
                return 0.8
        
        # Generic environmental data relevance
        if any(keyword in content for keyword in ["environmental", "climate", "atmospheric"]):
            return 0.5
        
        return 0.2  # Minimal relevance for any API
    
    def get_search_suggestions(self, feature_name: str) -> List[str]:
        """Get alternative search suggestions if initial search fails"""
        suggestions = [
            f"{feature_name} monitoring API Pakistan",
            f"{feature_name} historical data API free",
            f"{feature_name} REST API JSON South Asia",
            f"environmental data API {feature_name}",
            f"open data {feature_name} government Pakistan"
        ]
        
        return suggestions