import requests
import pandas as pd
import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import urllib.parse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class APIFetcher:
    def __init__(self, timeout_seconds: int = 15, max_retries: int = 3):
        """
        Initialize API Fetcher with retry logic and error handling
        
        Args:
            timeout_seconds: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Common headers
        self.session.headers.update({
            'User-Agent': 'DataEnrichmentAgent/1.0 (Environmental Monitoring)',
            'Accept': 'application/json, text/csv, application/xml',
            'Accept-Encoding': 'gzip, deflate'
        })
    
    def fetch_data(self, api_info: Dict[str, Any], feature_name: str, city: str, 
                   start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from an API endpoint
        
        Args:
            api_info: API configuration dictionary from LLM recommendations
            feature_name: Name of feature being fetched
            city: Target city
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with fetched data or None if failed
        """
        try:
            # Extract API configuration
            base_url = api_info.get("base_url", "")
            endpoint = api_info.get("endpoint", "")
            parameters = api_info.get("parameters", {})
            auth_type = api_info.get("authentication", "none")
            data_format = api_info.get("data_format", "json").lower()
            
            if not base_url:
                raise ValueError("No base_url provided in API configuration")
            
            # Build full URL
            full_url = self._build_url(base_url, endpoint)
            
            # Prepare parameters
            params = self._prepare_parameters(
                parameters, feature_name, city, start_date, end_date
            )
            
            # Setup authentication
            headers = self._setup_authentication(auth_type, api_info)
            
            # Make API request
            response = self._make_request(full_url, params, headers)
            
            if response is None:
                return None
            
            # Parse response based on format
            data = self._parse_response(response, data_format)
            
            if data is None:
                return None
            
            # Convert to DataFrame
            df = self._convert_to_dataframe(data, feature_name, api_info)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data from {api_info.get('name', 'Unknown API')}: {str(e)}")
            return None
    
    def _build_url(self, base_url: str, endpoint: str) -> str:
        """Build complete API URL"""
        if not endpoint:
            return base_url
        
        # Remove trailing slash from base_url and leading slash from endpoint
        base_url = base_url.rstrip('/')
        endpoint = endpoint.lstrip('/')
        
        return f"{base_url}/{endpoint}"
    
    def _prepare_parameters(self, template_params: Dict[str, Any], feature_name: str, 
                          city: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Prepare API parameters by substituting placeholders"""
        params = {}
        
        # Substitution mapping
        substitutions = {
            "{city}": city,
            "{feature_name}": feature_name,
            "{start_date}": start_date,
            "{end_date}": end_date,
            "{date_range_start}": start_date,
            "{date_range_end}": end_date,
            "start_date": start_date,
            "end_date": end_date,
            "city": city,
            "location": city,
            "q": city,
            "query": city
        }
        
        # Process template parameters
        for key, value in template_params.items():
            if isinstance(value, str):
                # Replace placeholders in string values
                processed_value = value
                for placeholder, replacement in substitutions.items():
                    processed_value = processed_value.replace(placeholder, replacement)
                params[key] = processed_value
            else:
                params[key] = value
        
        # Add common parameters if not specified
        if "key" not in params and "api_key" not in params:
            # Try to get API key from environment for common services
            api_key = self._get_api_key_for_service(template_params)
            if api_key:
                params["key"] = api_key
        
        return params
    
    def _get_api_key_for_service(self, api_info: Dict[str, Any]) -> Optional[str]:
        """Get API key from environment variables for known services"""
        import os
        
        base_url = api_info.get("base_url", "").lower()
        
        # Known API key environment variable mappings
        api_key_mappings = {
            "openweathermap.org": "OPENWEATHER_API_KEY",
            "weatherapi.com": "WEATHERAPI_KEY", 
            "airvisual.com": "AIRVISUAL_API_KEY",
            "waqi.info": "WAQI_API_KEY",
            "rapidapi.com": "RAPIDAPI_KEY"
        }
        
        for domain, env_var in api_key_mappings.items():
            if domain in base_url:
                return os.getenv(env_var)
        
        return None
    
    def _setup_authentication(self, auth_type: str, api_info: Dict[str, Any]) -> Dict[str, str]:
        """Setup authentication headers"""
        headers = {}
        
        if auth_type == "api_key":
            # API key authentication
            api_key = api_info.get("api_key") or self._get_api_key_for_service(api_info)
            if api_key:
                # Try common API key header formats
                headers["X-API-Key"] = api_key
                headers["Authorization"] = f"Bearer {api_key}"
        
        elif auth_type == "bearer":
            token = api_info.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        # Add any custom headers specified in API config
        custom_headers = api_info.get("headers", {})
        headers.update(custom_headers)
        
        return headers
    
    def _make_request(self, url: str, params: Dict[str, Any], 
                     headers: Dict[str, str]) -> Optional[requests.Response]:
        """Make HTTP request with error handling and retries"""
        try:
            # Add rate limiting delay
            time.sleep(0.1)  # 100ms delay to be respectful
            
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            
            # Check for HTTP errors
            if response.status_code == 401:
                print(f"Authentication failed for {url}")
                return None
            elif response.status_code == 403:
                print(f"Access forbidden for {url}")  
                return None
            elif response.status_code == 429:
                print(f"Rate limit exceeded for {url}")
                time.sleep(5)  # Wait before potentially retrying
                return None
            elif response.status_code >= 400:
                print(f"HTTP {response.status_code} error for {url}")
                return None
            
            return response
            
        except requests.exceptions.Timeout:
            print(f"Request timeout for {url}")
            return None
        except requests.exceptions.ConnectionError:
            print(f"Connection error for {url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {url}: {str(e)}")
            return None
    
    def _parse_response(self, response: requests.Response, 
                       data_format: str) -> Optional[Union[Dict, List, str]]:
        """Parse API response based on expected format"""
        try:
            if data_format == "json":
                return response.json()
            
            elif data_format == "csv":
                return response.text
            
            elif data_format == "xml":
                # Basic XML parsing - could be enhanced with xmltodict
                return response.text
            
            else:
                # Try JSON first, fallback to text
                try:
                    return response.json()
                except:
                    return response.text
                    
        except Exception as e:
            print(f"Failed to parse response: {str(e)}")
            return None
    
    def _convert_to_dataframe(self, data: Union[Dict, List, str], 
                            feature_name: str, api_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Convert parsed API response to pandas DataFrame"""
        try:
            if isinstance(data, str):
                # Handle CSV string data
                if data.strip().startswith('{') or data.strip().startswith('['):
                    # It's actually JSON in string format
                    data = json.loads(data)
                else:
                    # It's CSV data
                    from io import StringIO
                    return pd.read_csv(StringIO(data))
            
            if isinstance(data, dict):
                # Handle JSON object response
                df = self._extract_dataframe_from_json(data, feature_name, api_info)
                return df
            
            elif isinstance(data, list):
                # Handle JSON array response
                return pd.DataFrame(data)
            
            else:
                print(f"Unsupported data type for conversion: {type(data)}")
                return None
                
        except Exception as e:
            print(f"Failed to convert data to DataFrame: {str(e)}")
            return None
    
    def _extract_dataframe_from_json(self, json_data: Dict[str, Any], 
                                   feature_name: str, api_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Extract DataFrame from nested JSON response"""
        # Common patterns for nested data
        possible_data_keys = [
            'data', 'results', 'observations', 'records', 'list', 
            'weather', 'forecast', 'history', 'measurements', 'values'
        ]
        
        # Try to find the data array/list in nested JSON
        for key in possible_data_keys:
            if key in json_data:
                nested_data = json_data[key]
                if isinstance(nested_data, list):
                    return pd.DataFrame(nested_data)
                elif isinstance(nested_data, dict):
                    # Try to extract further
                    for sub_key in possible_data_keys:
                        if sub_key in nested_data and isinstance(nested_data[sub_key], list):
                            return pd.DataFrame(nested_data[sub_key])
        
        # If no nested structure found, try to flatten the JSON
        try:
            # Flatten the JSON and create DataFrame
            flattened_data = []
            
            def flatten_dict(d, parent_key='', sep='_'):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        # Handle list of dictionaries
                        for i, item in enumerate(v):
                            if isinstance(item, dict):
                                items.extend(flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)
            
            flattened = flatten_dict(json_data)
            return pd.DataFrame([flattened])
            
        except Exception as e:
            print(f"Failed to flatten JSON data: {str(e)}")
            return None
    
    def validate_api_response(self, df: pd.DataFrame, feature_name: str) -> bool:
        """Basic validation of API response DataFrame"""
        if df is None or df.empty:
            return False
        
        # Check if DataFrame has reasonable structure
        if len(df.columns) == 0:
            return False
        
        # Check for reasonable number of records
        if len(df) == 0:
            return False
        
        # Check for completely empty DataFrame
        if df.isnull().all().all():
            return False
        
        return True