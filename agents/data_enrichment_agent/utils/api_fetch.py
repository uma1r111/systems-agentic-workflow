# utils/api_fetch.py

import requests
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.data_validation import validate_enriched_data
from utils.storage import save_json

class APIFetcher:
    def __init__(
        self,
        city: str = None,
        max_workers: int = None,
        retries: int = None,
        output_path: str = None,
    ):
        # -----------------------------
        # Configuration (with defaults)
        # -----------------------------
        self.city = city or os.getenv("ENRICHMENT_CITY", "Karachi")
        self.max_workers = max_workers or int(os.getenv("ENRICHMENT_MAX_WORKERS", 4))
        self.retries = retries or int(os.getenv("ENRICHMENT_FETCH_RETRIES", 3))
        self.output_path = output_path or os.getenv("ENRICHMENT_OUTPUT_PATH", "outputs/enriched_data.json")

        # -----------------------------
        # Logger
        # -----------------------------
        self.logger = logging.getLogger("APIFetcher")
        self.logger.setLevel(logging.INFO)

    # -----------------------------
    # Fetch single API
    # -----------------------------
    def fetch_single_api(self, api_url, params=None):
        """Fetch data from a single API with retries and timeout."""
        for attempt in range(self.retries):
            try:
                response = requests.get(api_url, params=params, timeout=10)
                response.raise_for_status()
                self.logger.info(f"✅ Success: {api_url}")
                return response.json()
            except requests.RequestException as e:
                self.logger.warning(f"⚠️ Attempt {attempt+1}/{self.retries} failed for {api_url}: {e}")
        self.logger.error(f"❌ Failed to fetch data after {self.retries} attempts: {api_url}")
        return None

    # -----------------------------
    # Fetch from multiple APIs concurrently
    # -----------------------------
    def fetch_from_apis(self, api_list):
        """Fetch data from multiple APIs concurrently and save validated results."""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_api = {executor.submit(self.fetch_single_api, api): api for api in api_list}

            for future in as_completed(future_to_api):
                api_url = future_to_api[future]
                try:
                    data = future.result()
                    if data and validate_enriched_data(data):
                        results.append(data)
                except Exception as e:
                    self.logger.error(f"Unhandled exception for {api_url}: {e}")

        # Save only if we got results
        if results:
            save_json(results, self.output_path)
            self.logger.info(f"📂 Saved enriched data to {self.output_path}")
        else:
            self.logger.warning("⚠️ No valid enrichment data fetched.")

        return results


# -----------------------------
# Example Usage (manual run)
# -----------------------------
if __name__ == "__main__":
    fetcher = APIFetcher()
    api_candidates = [
        f"https://api.example.com/data?city={fetcher.city}",
        f"https://api.backup.com/data?city={fetcher.city}",
    ]
    fetcher.fetch_from_apis(api_candidates)
