import json
import os
import re
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import concurrent.futures as cf

import pandas as pd
import google.generativeai as genai

from utils.api_search import APISearcher
from utils.api_fetch import APIFetcher
from utils.data_validation import DataValidator
from utils.storage import DataStorage
from utils.logger import EnrichmentLogger

import os

ENRICHMENT_CITY = os.getenv("ENRICHMENT_CITY", "Karachi")
ENRICHMENT_MAX_BACKUP_APIS = int(os.getenv("ENRICHMENT_MAX_BACKUP_APIS", 3))

ENRICHMENT_START_DATE = os.getenv("ENRICHMENT_START_DATE", "2024-03-23")

ENRICHMENT_MAX_WORKERS = int(os.getenv("ENRICHMENT_MAX_WORKERS", 4))
ENRICHMENT_FETCH_RETRIES = int(os.getenv("ENRICHMENT_FETCH_RETRIES", 3))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", 0.4))
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", 1024))

DRIFT_OUTPUT_PATH = os.getenv("DRIFT_OUTPUT_PATH", "outputs/drift_analysis.json")
ENRICHMENT_OUTPUT_PATH = os.getenv("ENRICHMENT_OUTPUT_PATH", "outputs/enriched_data.json")


class DataEnrichmentAgent:
    """
    Orchestrates API discovery (LLM + Tavily), data fetching, validation, and storage
    for features suggested by the Drift Monitoring Agent.
    """

    def __init__(
        self,
        city: str = "Karachi",
        max_backup_apis: int = 2,
        *,
        start_date: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        fetch_timeout_seconds: int = 15,
        fetch_retries: int = 3,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Args:
            city: Target city for data collection.
            max_backup_apis: Max number of backup APIs to try per feature.
            start_date: Optional ISO date; defaults to ENRICHMENT_START_DATE env or "2025-03-23".
            model_name: Optional Gemini model name; defaults to GEMINI_MODEL env or "gemini-1.5-flash".
            temperature: Optional sampling temperature; defaults to GEMINI_TEMPERATURE env or 0.1.
            max_output_tokens: Optional token cap; defaults to GEMINI_MAX_TOKENS env or 800.
            fetch_timeout_seconds: Per-request timeout for API fetcher.
            fetch_retries: Retry attempts per API (exponential backoff).
            max_workers: Thread pool size for parallel feature processing. Defaults to min(4, len(features)).
        """
        self.city = city
        self.max_backup_apis = max_backup_apis
        self.start_date = (
            start_date
            or os.getenv("ENRICHMENT_START_DATE")
            or "2025-03-23"  # data starts Apr 1; Mar 23 keeps lagged features available
        )
        self.end_date = datetime.now().strftime("%Y-%m-%d")

        # Concurrency / fetch tuning
        self.fetch_retries = int(os.getenv("ENRICHMENT_FETCH_RETRIES", fetch_retries))
        self.max_workers = (
            int(os.getenv("ENRICHMENT_MAX_WORKERS", max_workers)) if max_workers is not None else max_workers
        )

        # Resolve repo-relative paths robustly
        self.base_dir = Path(__file__).resolve().parent
        self.prompts_dir = self.base_dir / "prompts"
        self.outputs_dir = self.base_dir / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Utilities
        self.api_searcher = APISearcher()
        self.api_fetcher = APIFetcher(timeout_seconds=fetch_timeout_seconds)
        self.data_validator = DataValidator()
        self.data_storage = DataStorage()
        self.logger = EnrichmentLogger()

        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        _model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.model = genai.GenerativeModel(_model_name)
        self._temperature = float(os.getenv("GEMINI_TEMPERATURE", temperature if temperature is not None else 0.1))
        self._max_output_tokens = int(os.getenv("GEMINI_MAX_TOKENS", max_output_tokens if max_output_tokens is not None else 800))

    # ---------------------------
    # Drift output loading
    # ---------------------------
    def load_drift_results(self, drift_output_path: str) -> Dict[str, Any]:
        """Load results from Drift Monitoring Agent."""
        path = Path(drift_output_path)
        if not path.exists():
            raise FileNotFoundError(f"Drift analysis results not found at {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ---------------------------
    # API discovery via LLM + Tavily
    # ---------------------------
    def _read_prompt_template(self) -> str:
        prompt_template_path = self.prompts_dir / "api_discovery_prompt.txt"
        if not prompt_template_path.exists():
            raise FileNotFoundError(f"Prompt template not found at {prompt_template_path}")
        return prompt_template_path.read_text(encoding="utf-8")

    def _parse_llm_json(self, raw_text: str) -> Dict[str, Any]:
        """Parse LLM output as JSON, with a regex fallback to the first JSON object found."""
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", raw_text)
            if match:
                return json.loads(match.group(0))
            raise

    def discover_apis_for_feature(self, feature_name: str) -> Dict[str, Any]:
        """Use LLM + Tavily to discover APIs for a specific feature."""
        prompt_template = self._read_prompt_template()

        prompt = (
            prompt_template
            .replace("{feature_name}", feature_name)
            .replace("{city}", self.city)
            .replace("{date_range}", f"{self.start_date} to {self.end_date}")
        )

        # First, use Tavily to search for APIs
        search_results = self.api_searcher.search_apis(feature_name, self.city)
        full_prompt = f"{prompt}\n\nTavily Search Results:\n{json.dumps(search_results, indent=2)}"

        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self._temperature,
                    max_output_tokens=self._max_output_tokens,
                    response_mime_type="application/json",
                ),
            )

            raw_text = getattr(response, "text", None)
            if not raw_text:
                # Fallback to candidates structure
                raw_text = response.candidates[0].content.parts[0].text

            api_recommendations = self._parse_llm_json(raw_text)

            # Normalize minimal structure expected downstream
            api_recommendations.setdefault("primary_api", None)
            api_recommendations.setdefault("backup_apis", [])
            api_recommendations.setdefault("documentation_links", [])
            api_recommendations.setdefault("confidence", "unknown")
            api_recommendations.setdefault("reasoning", "")

            return api_recommendations

        except Exception as e:
            self.logger.log_error(f"LLM API discovery failed for {feature_name}: {str(e)}")
            return {
                "primary_api": None,
                "backup_apis": [],
                "documentation_links": [],
                "confidence": "low",
                "reasoning": f"API discovery failed: {str(e)}",
            }

    # ---------------------------
    # Fetch + validate single feature
    # ---------------------------
    def _retry_fetch(self, api_info: Dict[str, Any], feature_name: str) -> Optional[pd.DataFrame]:
        """Fetch with retries and jittered exponential backoff."""
        last_err: Optional[Exception] = None
        for attempt in range(self.fetch_retries):
            try:
                return self.api_fetcher.fetch_data(
                    api_info,
                    feature_name,
                    self.city,
                    self.start_date,
                    self.end_date,
                )
            except Exception as e:
                last_err = e
                wait = (2 ** attempt) + random.random() * 0.5
                self.logger.log_warning(
                    f"Fetch attempt {attempt + 1}/{self.fetch_retries} failed for {feature_name} via {api_info.get('name', 'Unknown')}: {e}. Retrying in ~{wait:.1f}s"
                )
                time.sleep(wait)
        if last_err:
            raise last_err
        return None

    def fetch_feature_data(self, feature_name: str, api_config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Fetch data for a single feature using primary + backup APIs."""
        apis_to_try: List[Dict[str, Any]] = []

        # Add primary API if available
        primary = api_config.get("primary_api")
        if primary:
            apis_to_try.append(primary)

        # Add backup APIs (even if primary is missing)
        backup_apis = (api_config.get("backup_apis") or [])[: self.max_backup_apis]
        apis_to_try.extend(backup_apis)

        if not apis_to_try:
            self.logger.log_warning(f"No APIs to try for feature '{feature_name}' (no primary or backups).")
            return None

        for i, api_info in enumerate(apis_to_try):
            api_type = "primary" if (primary and i == 0) else ("backup_1" if i == 0 else f"backup_{i}")
            try:
                self.logger.log_info(
                    f"Attempting {api_type} API for {feature_name}: {api_info.get('name', 'Unknown')}"
                )

                raw_data = self._retry_fetch(api_info, feature_name)

                if raw_data is not None and len(raw_data) > 0:
                    # Validate/clean
                    validated_data = self.data_validator.validate_and_clean(raw_data, feature_name)

                    if validated_data is not None and len(validated_data) > 0:
                        self.logger.log_success(
                            feature_name=feature_name,
                            api_used=api_info.get("name", "Unknown"),
                            api_type=api_type,
                            record_count=len(validated_data),
                        )
                        return validated_data
                    else:
                        self.logger.log_warning(
                            f"Data validation produced no usable records for {feature_name} using {api_type}"
                        )
                else:
                    self.logger.log_warning(
                        f"No data returned from {api_type} for {feature_name}"
                    )

            except Exception as e:
                self.logger.log_error(f"{api_type} API failed for {feature_name}: {str(e)}")
                continue

        # All APIs failed
        self.logger.log_failure(
            feature_name=feature_name, reason="All APIs failed or returned invalid data"
        )
        return None

    # ---------------------------
    # One-feature end-to-end
    # ---------------------------
    def process_single_feature(self, feature_name: str) -> Dict[str, Any]:
        """Process a single feature: discover APIs, fetch data, validate, and store."""
        self.logger.log_info(f"Processing feature: {feature_name}")

        # Step 1: Discover APIs
        api_config = self.discover_apis_for_feature(feature_name)

        if not api_config.get("primary_api") and not api_config.get("backup_apis"):
            return {
                "feature_name": feature_name,
                "status": "failed",
                "reason": "No suitable APIs found (no primary or backups)",
                "data_file": None,
            }

        # Step 2: Fetch and validate data
        feature_data = self.fetch_feature_data(feature_name, api_config)

        if feature_data is None or len(feature_data) == 0:
            return {
                "feature_name": feature_name,
                "status": "failed",
                "reason": "Data fetch failed for all APIs or produced empty data after validation",
                "data_file": None,
            }

        # Step 3: Store data (keep existing save behavior, add timestamped copy for versioning)
        try:
            output_file = self.data_storage.save_feature_data(
                feature_data,
                feature_name,
                self.city,
            )

            # Add timestamped version alongside original to avoid accidental overwrite loss
            try:
                original = Path(output_file)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                versioned = original.with_name(f"{original.stem}_{ts}{original.suffix}")
                if original.exists():
                    import shutil

                    shutil.copyfile(original, versioned)
                    final_path = str(versioned)
                else:
                    final_path = str(original)
            except Exception:
                # If versioning fails, fall back to the original path
                final_path = output_file

            return {
                "feature_name": feature_name,
                "status": "success",
                "reason": f"Successfully fetched {len(feature_data)} records",
                "data_file": final_path,
                "record_count": len(feature_data),
            }

        except Exception as e:
            self.logger.log_error(f"Failed to save data for {feature_name}: {str(e)}")
            return {
                "feature_name": feature_name,
                "status": "failed",
                "reason": f"Data storage failed: {str(e)}",
                "data_file": None,
            }

    # ---------------------------
    # Orchestration
    # ---------------------------
    def run_enrichment(self, drift_output_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all features from drift analysis.

        Args:
            drift_output_path: Path to drift monitoring agent output JSON.
            output_path: Optional path to save enrichment summary/results JSON.
        """
        self.logger.log_info("Starting data enrichment process")

        # Load drift analysis results
        try:
            drift_results = self.load_drift_results(drift_output_path)
            features_to_add: List[str] = drift_results.get("features_to_add", [])

            if not features_to_add:
                self.logger.log_info("No features to add - drift analysis found no significant changes")
                return {
                    "enrichment_triggered": False,
                    "reason": "No features recommended by drift analysis",
                    "processed_features": [],
                    "summary": {"total": 0, "success": 0, "failed": 0, "success_rate": "0%"},
                }

            preview = features_to_add[:5]
            self.logger.log_info(
                f"Processing {len(features_to_add)} features (preview: {preview}{'...' if len(features_to_add) > 5 else ''})"
            )

        except Exception as e:
            self.logger.log_error(f"Failed to load drift results: {str(e)}")
            return {
                "enrichment_triggered": False,
                "reason": f"Failed to load drift results: {str(e)}",
                "processed_features": [],
                "summary": {"total": 0, "success": 0, "failed": 0, "success_rate": "0%"},
            }

        # Decide workers only now that we know how many features we have
        workers = self.max_workers if self.max_workers is not None else max(1, min(4, len(features_to_add)))

        processed_features: List[Dict[str, Any]] = []
        success_count = 0

        # Parallelize per-feature processing (I/O-bound)
        with cf.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.process_single_feature, feat): feat for feat in features_to_add}
            for fut in cf.as_completed(futures):
                feature_name = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    self.logger.log_error(f"Unexpected error processing {feature_name}: {str(e)}")
                    result = {
                        "feature_name": feature_name,
                        "status": "failed",
                        "reason": f"Unexpected error: {str(e)}",
                        "data_file": None,
                    }
                processed_features.append(result)
                if result.get("status") == "success":
                    success_count += 1

        # Create summary
        total = len(features_to_add)
        summary = {
            "total": total,
            "success": success_count,
            "failed": total - success_count,
            "success_rate": f"{(success_count / total * 100):.1f}%" if total else "0%",
        }

        # Compile final output
        output = {
            "enrichment_triggered": True,
            "city": self.city,
            "date_range": f"{self.start_date} to {self.end_date}",
            "processed_features": processed_features,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }

        # Save output if path provided
        if output_path:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, default=str)

        self.logger.log_info(f"Enrichment complete. Success rate: {summary['success_rate']}")
        return output


if __name__ == "__main__":
    # Example usage
    agent = DataEnrichmentAgent(
        city=os.getenv("ENRICHMENT_CITY", "Karachi"),
        max_backup_apis=int(os.getenv("ENRICHMENT_MAX_BACKUP_APIS", 2)),
    )

    result = agent.run_enrichment(
        drift_output_path=os.getenv(
            "DRIFT_OUTPUT_PATH",
            "agents/drift_monitoring_agent/outputs/drift_analysis.json",
        ),
        output_path=os.getenv(
            "ENRICHMENT_OUTPUT_PATH",
            "agents/data_enrichment_agent/outputs/enrichment_results.json",
        ),
    )

    print("Data Enrichment Complete!")
    print(f"Enrichment triggered: {result['enrichment_triggered']}")
    print(f"Summary: {result.get('summary', {})}")

    for feature_result in result.get("processed_features", []):
        print(f"  {feature_result['feature_name']}: {feature_result['status']} - {feature_result['reason']}")
