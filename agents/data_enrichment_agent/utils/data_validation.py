import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import re
from dateutil import parser
import warnings
warnings.filterwarnings('ignore')

class DataValidator:
    def __init__(self):
        """Initialize Data Validator with validation rules and unit conversions"""
        
        # Define expected value ranges for different feature types
        self.feature_ranges = {
            # Weather features
            "temperature": {"min": -50, "max": 60, "unit": "celsius"},
            "humidity": {"min": 0, "max": 100, "unit": "percent"},
            "wind_speed": {"min": 0, "max": 200, "unit": "kph"},
            "precipitation": {"min": 0, "max": 1000, "unit": "mm"},
            "pressure": {"min": 800, "max": 1100, "unit": "hPa"},
            "visibility": {"min": 0, "max": 50, "unit": "km"},
            
            # Air quality features (µg/m³ unless specified)
            "pm2_5": {"min": 0, "max": 1000, "unit": "µg/m³"},
            "pm10": {"min": 0, "max": 2000, "unit": "µg/m³"},
            "no2": {"min": 0, "max": 500, "unit": "µg/m³"},
            "so2": {"min": 0, "max": 1000, "unit": "µg/m³"},
            "co": {"min": 0, "max": 50, "unit": "mg/m³"},
            "o3": {"min": 0, "max": 500, "unit": "µg/m³"},
            "aqi": {"min": 0, "max": 500, "unit": "index"},
            
            # Derived features
            "uv_index": {"min": 0, "max": 15, "unit": "index"},
            "pollen": {"min": 0, "max": 1000, "unit": "grains/m³"}
        }
        
        # Common column name mappings for standardization
        self.column_mappings = {
            # Temperature variants
            "temp": "temperature", "temperature_c": "temperature", "temp_c": "temperature",
            "temperature_celsius": "temperature", "air_temp": "temperature",
            
            # Humidity variants
            "humidity_percent": "humidity", "relative_humidity": "humidity", "rh": "humidity",
            
            # Wind speed variants
            "wind": "wind_speed", "windspeed": "wind_speed", "wind_speed_kph": "wind_speed",
            "wind_velocity": "wind_speed", "wspd": "wind_speed",
            
            # Precipitation variants
            "rain": "precipitation", "rainfall": "precipitation", "precip": "precipitation",
            "precipitation_mm": "precipitation",
            
            # Air quality variants
            "pm25": "pm2_5", "pm_25": "pm2_5", "pm2.5": "pm2_5",
            "particulate_matter_25": "pm2_5", "fine_particles": "pm2_5",
            "particulate_matter_10": "pm10", "coarse_particles": "pm10",
            "nitrogen_dioxide": "no2", "sulfur_dioxide": "so2", "carbon_monoxide": "co", "ozone": "o3",
            
            # DateTime variants
            "date": "datetime", "timestamp": "datetime", "time": "datetime",
            "date_time": "datetime", "dt": "datetime", "observation_time": "datetime"
        }
        
        # Unit conversion factors
        self.unit_conversions = {
            # Temperature conversions
            "fahrenheit_to_celsius": lambda f: (f - 32) * 5/9,
            "kelvin_to_celsius": lambda k: k - 273.15,
            
            # Speed conversions
            "mph_to_kph": lambda mph: mph * 1.60934,
            "ms_to_kph": lambda ms: ms * 3.6,
            "knots_to_kph": lambda knots: knots * 1.852,
            
            # Pressure conversions
            "mmhg_to_hpa": lambda mmhg: mmhg * 1.33322,
            "inhg_to_hpa": lambda inhg: inhg * 33.8639,
            "psi_to_hpa": lambda psi: psi * 68.9476,
            
            # Air quality conversions (ppm to µg/m³ at standard conditions)
            "no2_ppm_to_ugm3": lambda ppm: ppm * 1880,
            "so2_ppm_to_ugm3": lambda ppm: ppm * 2620,
            "co_ppm_to_mgm3": lambda ppm: ppm * 1.145,
            "o3_ppm_to_ugm3": lambda ppm: ppm * 1960
        }
    
    def validate_and_clean(self, df: pd.DataFrame, feature_name: str) -> Optional[pd.DataFrame]:
        """
        Main validation and cleaning method
        
        Args:
            df: Raw DataFrame from API
            feature_name: Name of the feature being validated
            
        Returns:
            Cleaned and validated DataFrame or None if validation fails
        """
        if df is None or df.empty:
            return None
        
        try:
            # Step 1: Standardize column names
            df_clean = self._standardize_columns(df.copy())
            
            # Step 2: Identify and parse datetime column
            df_clean = self._parse_datetime_column(df_clean)
            
            # Step 3: Identify target feature column
            feature_column = self._identify_feature_column(df_clean, feature_name)
            if not feature_column:
                print(f"Could not identify feature column for {feature_name}")
                return None
            
            # Step 4: Clean and convert feature values
            df_clean = self._clean_feature_values(df_clean, feature_column, feature_name)
            
            # Step 5: Validate data quality
            if not self._validate_data_quality(df_clean, feature_column, feature_name):
                print(f"Data quality validation failed for {feature_name}")
                return None
            
            # Step 6: Sort by datetime and remove duplicates
            df_clean = self._finalize_dataframe(df_clean, feature_column, feature_name)
            
            return df_clean
            
        except Exception as e:
            print(f"Error during data validation for {feature_name}: {str(e)}")
            return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using predefined mappings"""
        # Convert all column names to lowercase and remove spaces/special chars
        df.columns = df.columns.str.lower().str.replace(r'[^\w]', '_', regex=True)
        
        # Apply column mappings
        rename_dict = {}
        for col in df.columns:
            for old_name, new_name in self.column_mappings.items():
                if old_name in col:
                    rename_dict[col] = new_name
                    break
        
        if rename_dict:
            df.rename(columns=rename_dict, inplace=True)
        
        return df
    
    def _parse_datetime_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify and parse datetime column"""
        datetime_candidates = ['datetime', 'date', 'timestamp', 'time']
        
        datetime_col = None
        for col in datetime_candidates:
            if col in df.columns:
                datetime_col = col
                break
        
        # If no standard datetime column found, look for date-like columns
        if not datetime_col:
            for col in df.columns:
                if any(keyword in col for keyword in ['date', 'time', 'dt']):
                    datetime_col = col
                    break
        
        # Try to parse datetime
        if datetime_col:
            try:
                df[datetime_col] = pd.to_datetime(df[datetime_col], infer_datetime_format=True, errors='coerce')
                # Set as index if successfully parsed
                if not df[datetime_col].isnull().all():
                    df.set_index(datetime_col, inplace=True)
                    df.index.name = 'datetime'
            except Exception as e:
                print(f"Failed to parse datetime column {datetime_col}: {str(e)}")
        
        return df
    
    def _identify_feature_column(self, df: pd.DataFrame, feature_name: str) -> Optional[str]:
        """Identify the column containing the target feature data"""
        feature_lower = feature_name.lower()
        
        # Direct match
        if feature_lower in df.columns:
            return feature_lower
        
        # Partial matches
        for col in df.columns:
            if feature_lower in col or col in feature_lower:
                return col
        
        # Try feature-specific patterns
        feature_patterns = {
            "temperature": ["temp", "temperature", "air_temp"],
            "humidity": ["humidity", "rh", "relative"],
            "wind_speed": ["wind", "wspd", "speed"],
            "precipitation": ["precip", "rain", "rainfall"],
            "pm2_5": ["pm25", "pm_25", "pm2", "fine"],
            "pm10": ["pm10", "coarse"],
            "no2": ["no2", "nitrogen"],
            "so2": ["so2", "sulfur"],
            "co": ["co", "carbon"],
            "o3": ["o3", "ozone"],
            "aqi": ["aqi", "index", "quality"]
        }
        
        patterns = feature_patterns.get(feature_lower, [feature_lower])
        for pattern in patterns:
            for col in df.columns:
                if pattern in col:
                    return col
        
        # If still not found, return the first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]
        
        return None
    
    def _clean_feature_values(self, df: pd.DataFrame, feature_col: str, feature_name: str) -> pd.DataFrame:
        """Clean and convert feature values"""
        if feature_col not in df.columns:
            return df
        
        # Convert to numeric, replacing non-numeric values with NaN
        df[feature_col] = pd.to_numeric(df[feature_col], errors='coerce')
        
        # Apply unit conversions if needed
        df = self._apply_unit_conversions(df, feature_col, feature_name)
        
        # Remove extreme outliers
        df = self._remove_outliers(df, feature_col, feature_name)
        
        # Fill small gaps in data
        df = self._fill_data_gaps(df, feature_col)
        
        return df
    
    def _apply_unit_conversions(self, df: pd.DataFrame, feature_col: str, feature_name: str) -> pd.DataFrame:
        """Apply unit conversions based on detected units"""
        if feature_col not in df.columns:
            return df
        
        values = df[feature_col].dropna()
        if len(values) == 0:
            return df
        
        feature_lower = feature_name.lower()
        
        # Temperature conversions
        if "temperature" in feature_lower:
            # Detect if values are in Fahrenheit (typically > 50 for weather data)
            if values.mean() > 50 and values.max() > 80:
                print(f"Converting temperature from Fahrenheit to Celsius")
                df[feature_col] = self.unit_conversions["fahrenheit_to_celsius"](df[feature_col])
            # Detect if values are in Kelvin (typically > 250)
            elif values.mean() > 250:
                print(f"Converting temperature from Kelvin to Celsius")
                df[feature_col] = self.unit_conversions["kelvin_to_celsius"](df[feature_col])
        
        # Wind speed conversions
        elif "wind" in feature_lower:
            # If values are very small (likely m/s), convert to km/h
            if values.mean() < 20 and values.max() < 50:
                print(f"Converting wind speed from m/s to km/h")
                df[feature_col] = self.unit_conversions["ms_to_kph"](df[feature_col])
        
        # Air quality unit detection (if values are very small, likely in ppm)
        elif feature_lower in ["no2", "so2", "co", "o3"]:
            if values.mean() < 1 and values.max() < 10:
                print(f"Converting {feature_name} from ppm to µg/m³")
                if feature_lower == "no2":
                    df[feature_col] = self.unit_conversions["no2_ppm_to_ugm3"](df[feature_col])
                elif feature_lower == "so2":
                    df[feature_col] = self.unit_conversions["so2_ppm_to_ugm3"](df[feature_col])
                elif feature_lower == "co":
                    df[feature_col] = self.unit_conversions["co_ppm_to_mgm3"](df[feature_col])
                elif feature_lower == "o3":
                    df[feature_col] = self.unit_conversions["o3_ppm_to_ugm3"](df[feature_col])
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, feature_col: str, feature_name: str) -> pd.DataFrame:
        """Remove extreme outliers based on feature-specific ranges"""
        if feature_col not in df.columns:
            return df
        
        feature_lower = feature_name.lower()
        
        # Get expected range for this feature type
        expected_range = None
        for range_key, range_info in self.feature_ranges.items():
            if range_key in feature_lower or feature_lower in range_key:
                expected_range = range_info
                break
        
        if expected_range:
            # Remove values outside expected range
            min_val, max_val = expected_range["min"], expected_range["max"]
            outlier_mask = (df[feature_col] < min_val) | (df[feature_col] > max_val)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                print(f"Removing {outlier_count} outliers outside range [{min_val}, {max_val}] for {feature_name}")
                df.loc[outlier_mask, feature_col] = np.nan
        
        else:
            # Use statistical outlier removal (IQR method)
            Q1 = df[feature_col].quantile(0.25)
            Q3 = df[feature_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outlier_mask = (df[feature_col] < lower_bound) | (df[feature_col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                print(f"Removing {outlier_count} statistical outliers for {feature_name}")
                df.loc[outlier_mask, feature_col] = np.nan
        
        return df
    
    def _fill_data_gaps(self, df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
        """Fill small gaps in data using interpolation"""
        if feature_col not in df.columns:
            return df
        
        # Only interpolate small gaps (up to 3 consecutive missing values)
        df[feature_col] = df[feature_col].interpolate(method='linear', limit=3, limit_direction='both')
        
        # Forward/backward fill for remaining small gaps
        df[feature_col] = df[feature_col].fillna(method='ffill', limit=1)
        df[feature_col] = df[feature_col].fillna(method='bfill', limit=1)
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame, feature_col: str, feature_name: str) -> bool:
        """Validate overall data quality"""
        if feature_col not in df.columns:
            return False
        
        # Check if we have any valid data
        valid_data = df[feature_col].dropna()
        if len(valid_data) == 0:
            print(f"No valid data found for {feature_name}")
            return False
        
        # Check data completeness (at least 50% non-null)
        completeness = len(valid_data) / len(df)
        if completeness < 0.5:
            print(f"Data completeness too low: {completeness:.2%} for {feature_name}")
            return False
        
        # Check for reasonable variance (not all same values)
        if valid_data.nunique() <= 1:
            print(f"No variance in data for {feature_name}")
            return False
        
        # Check temporal ordering if datetime index exists
        if isinstance(df.index, pd.DatetimeIndex):
            if not df.index.is_monotonic_increasing:
                print(f"Datetime index not properly ordered for {feature_name}")
                # Try to sort
                df.sort_index(inplace=True)
        
        return True
    
    def _finalize_dataframe(self, df: pd.DataFrame, feature_col: str, feature_name: str) -> pd.DataFrame:
        """Final cleanup and standardization of DataFrame"""
        # Remove rows where feature column is null
        df_final = df[df[feature_col].notna()].copy()
        
        # Rename feature column to standardized name
        df_final.rename(columns={feature_col: feature_name}, inplace=True)
        
        # Remove duplicate timestamps if datetime index exists
        if isinstance(df_final.index, pd.DatetimeIndex):
            df_final = df_final[~df_final.index.duplicated(keep='first')]
            df_final.sort_index(inplace=True)
        
        # Keep only essential columns (datetime index + feature + any metadata)
        essential_cols = [feature_name]
        
        # Add useful metadata columns if they exist
        metadata_cols = ['location', 'station_id', 'quality_flag', 'source']
        for col in metadata_cols:
            if col in df_final.columns:
                essential_cols.append(col)
        
        df_final = df_final[essential_cols]
        
        return df_final
    
    def get_validation_summary(self, df: pd.DataFrame, feature_name: str) -> Dict[str, Any]:
        """Generate validation summary statistics"""
        if df is None or df.empty:
            return {"status": "failed", "reason": "Empty or null DataFrame"}
        
        feature_col = feature_name if feature_name in df.columns else df.columns[0]
        
        summary = {
            "status": "success",
            "feature_name": feature_name,
            "total_records": len(df),
            "valid_records": df[feature_col].count(),
            "completeness": f"{(df[feature_col].count() / len(df) * 100):.1f}%",
            "date_range": {
                "start": str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) else "N/A",
                "end": str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else "N/A"
            },
            "statistics": {
                "mean": round(df[feature_col].mean(), 3),
                "std": round(df[feature_col].std(), 3),
                "min": round(df[feature_col].min(), 3),
                "max": round(df[feature_col].max(), 3),
                "median": round(df[feature_col].median(), 3)
            }
        }
        
        return summary