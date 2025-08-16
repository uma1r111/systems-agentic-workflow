import os
import requests
from datetime import datetime
from typing import Dict, Any, List


class APISearcher:
    def __init__(self, max_results: int = 10, search_depth: str = "basic"):
        """Initialize Tavily API Searcher"""
        self.api_key = os.getenv("TAVILY_API_KEY", "your_api_key_here")  # fallback for direct use
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not set")
        self.base_url = "https://api.tavily.com/search"
        self.max_results = max_results
        self.search_depth = search_depth

    # -----------------------------
    # Query construction
    # -----------------------------
    def _construct_query(self, feature: str, city: str) -> str:
        """Generate search query for Tavily"""
        return f"{feature} API {city} Pakistan OR historical data OR JSON REST OR free public API"

    # -----------------------------
    # Main search
    # -----------------------------
    def search(self, feature: str, city: str) -> Dict[str, Any]:
        query = self._construct_query(feature, city)
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": self.search_depth,
            "include_answer": True,
            "max_results": self.max_results,
            "include_domains": [
                "api.openweathermap.org",
                "api.weatherapi.com",
                "api.airvisual.com",
                "api.waqi.info",
                "rapidapi.com",
                "opendata.gov.pk",
                "pmd.gov.pk"
            ]
        }

        try:
            r = requests.post(self.base_url, json=payload, timeout=20)
            r.raise_for_status()
            return self._process_results(r.json(), feature, city, query)
        except Exception as e:
            return {"status": "error", "error": str(e), "query_used": query, "results": []}

    # -----------------------------
    # Processing results (Hybrid)
    # -----------------------------
    def _process_results(self, raw: Dict[str, Any], feature: str, city: str, query: str) -> Dict[str, Any]:
        results = []
        for res in raw.get("results", []):
            if self._looks_like_api(res, feature, city):
                score = self._score_api_candidate(res, feature, city)
                results.append({
                    "title": res.get("title", ""),
                    "url": res.get("url", ""),
                    "score": score,
                })

        # Sort by score (descending)
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return {
            "status": "success",
            "query_used": query,
            "total_found": len(raw.get("results", [])),
            "api_candidates": results[:8],  # Top 8
            "timestamp": datetime.now().isoformat(),
            "tavily_answer": raw.get("answer", "")
        }

    # -----------------------------
    # Simple API relevance filter
    # -----------------------------
    def _looks_like_api(self, res: Dict[str, Any], feature: str, city: str) -> bool:
        text = (res.get("title", "") + " " + res.get("content", "")).lower()
        url = res.get("url", "").lower()

        # Must contain API indicators
        if not any(k in text for k in ["api", "endpoint", "json", "docs"]):
            return False

        # Discard blogs/news
        if any(k in text for k in ["blog", "news", "forum", "article"]):
            return False

        # Loose relevance check
        return feature.lower() in text or city.lower() in text or "pakistan" in text

    # -----------------------------
    # Scoring system
    # -----------------------------
    def _score_api_candidate(self, res: Dict[str, Any], feature: str, city: str) -> int:
        """Assign a score based on feature/city/API keywords relevance"""
        text = (res.get("title", "") + " " + res.get("content", "")).lower()
        url = res.get("url", "").lower()
        score = 0

        # Base keyword boosts
        if "api" in text: score += 5
        if "json" in text: score += 3
        if "endpoint" in text: score += 2
        if "docs" in text: score += 2

        # Feature and city relevance
        if feature.lower() in text: score += 4
        if city.lower() in text: score += 3
        if "pakistan" in text: score += 2

        # Trusted domains get a boost
        trusted_domains = [
            "openweathermap.org",
            "weatherapi.com",
            "airvisual.com",
            "waqi.info",
            "rapidapi.com",
            "opendata.gov.pk",
            "pmd.gov.pk"
        ]
        if any(domain in url for domain in trusted_domains):
            score += 5

        return score
