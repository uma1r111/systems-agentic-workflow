import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from scipy import stats
from typing import Dict, List, Any
import os
from pathlib import Path

class DriftMonitoringAgent:
    def __init__(self, drift_threshold: float = 0.15):
        """
        Initialize Drift Monitoring Agent
        
        Args:
            drift_threshold: Threshold for detecting significant drift (15% by default)
        """
        self.drift_threshold = drift_threshold
        self.raw_features = [
            'temp_C', 'humidity_%', 'windspeed_kph', 'precip_mm',
            'pm10', 'pm2_5', 'co', 'no2', 'so2', 'o3'
        ]
        self.engineered_features = {
            'logged': ['log_co', 'log_pm2_5', 'log_pm10', 'log_precip_mm', 
                      'log_so2', 'log_windspeed_kph', 'log_no2', 'log_o3'],
            'scaled': ['scaled_temp_C', 'scaled_humidity_%', 'scaled_log_windspeed_kph',
                      'scaled_log_pm2_5', 'scaled_log_pm10', 'scaled_log_precip_mm',
                      'scaled_log_co', 'scaled_log_no2', 'scaled_log_so2', 'scaled_o3'],
            'lags': ['aqi_us_lag1', 'aqi_us_lag12', 'aqi_us_lag24',
                    'scaled_log_pm2_5_lag1', 'scaled_log_pm2_5_lag12', 'scaled_log_pm2_5_lag24',
                    'scaled_log_pm10_lag1', 'scaled_log_pm10_lag12', 'scaled_log_pm10_lag24'],
            'time_encoding': ['hour_sin', 'hour_cos', 'hour', 'day_of_week', 'is_weekend'],
            'interactions': ['log_pm2_5_scaled_log_windspeed_kph', 'scaled_temp_C_scaled_o3',
                           'scaled_temp_C_scaled_log_windspeed_kph']
        }
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and prepare data with datetime index"""
        df = pd.read_csv(csv_path)
        
        # Convert datetime column (adjust column name as needed)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        return df.sort_index()
    
    def split_time_periods(self, df: pd.DataFrame) -> tuple:
        """Split data into recent (7 days) and baseline (90 days before recent)"""
        end_date = df.index.max()
        recent_start = end_date - timedelta(days=7)
        baseline_end = recent_start
        baseline_start = baseline_end - timedelta(days=90)
        
        recent_data = df[recent_start:end_date]
        baseline_data = df[baseline_start:baseline_end]
        
        return recent_data, baseline_data
    
    def calculate_drift_metrics(self, recent: pd.Series, baseline: pd.Series) -> Dict[str, float]:
        """Calculate drift metrics for a single feature"""
        if len(recent) == 0 or len(baseline) == 0:
            return {"mean_shift": 0, "variance_shift": 0, "ks_statistic": 0}
        
        recent_clean = recent.dropna()
        baseline_clean = baseline.dropna()
        
        if len(recent_clean) == 0 or len(baseline_clean) == 0:
            return {"mean_shift": 0, "variance_shift": 0, "ks_statistic": 0}
        
        baseline_mean = baseline_clean.mean()
        recent_mean = recent_clean.mean()
        mean_shift = abs((recent_mean - baseline_mean) / baseline_mean) if baseline_mean != 0 else 0
        
        baseline_var = baseline_clean.var()
        recent_var = recent_clean.var()
        variance_shift = abs((recent_var - baseline_var) / baseline_var) if baseline_var != 0 else 0
        
        ks_stat, _ = stats.ks_2samp(baseline_clean, recent_clean)
        
        return {
            "mean_shift": mean_shift,
            "variance_shift": variance_shift,
            "ks_statistic": ks_stat
        }
    
    def detect_seasonal_shifts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal shifts in all features"""
        recent_data, baseline_data = self.split_time_periods(df)
        
        drift_results = {}
        significant_drifts = []
        
        all_features = (self.raw_features + 
                       self.engineered_features['logged'] +
                       self.engineered_features['scaled'] +
                       self.engineered_features['lags'] +
                       self.engineered_features['time_encoding'] +
                       self.engineered_features['interactions'])
        
        for feature in all_features:
            if feature in df.columns:
                metrics = self.calculate_drift_metrics(
                    recent_data[feature], 
                    baseline_data[feature]
                )
                drift_results[feature] = metrics
                
                if (metrics['mean_shift'] > self.drift_threshold or 
                    metrics['variance_shift'] > self.drift_threshold or
                    metrics['ks_statistic'] > 0.2):
                    significant_drifts.append({
                        'feature': feature,
                        'drift_type': self._classify_drift(metrics),
                        'metrics': metrics
                    })
        
        return {
            'all_metrics': drift_results,
            'significant_drifts': significant_drifts,
            'drift_detected': len(significant_drifts) > 0
        }
    
    def _classify_drift(self, metrics: Dict[str, float]) -> str:
        if metrics['mean_shift'] > self.drift_threshold:
            return "mean_shift"
        elif metrics['variance_shift'] > self.drift_threshold:
            return "variance_shift"
        elif metrics['ks_statistic'] > 0.2:
            return "distribution_shift"
        return "no_drift"
    
    def generate_feature_recommendations(self, drift_results: Dict[str, Any]) -> List[str]:
        recommendations = []
        significant_drifts = drift_results['significant_drifts']
        
        pollutant_drifts = [d for d in significant_drifts 
                           if any(pol in d['feature'] for pol in ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3'])]
        weather_drifts = [d for d in significant_drifts 
                         if any(weather in d['feature'] for weather in ['temp', 'humidity', 'wind', 'precip'])]
        
        if pollutant_drifts:
            recommendations.extend([
                "uv_index", "pm1", "visibility_km", "air_pressure_hpa"
            ])
        if weather_drifts:
            recommendations.extend([
                "dew_point_C", "heat_index", "wind_direction", "cloud_cover_%"
            ])
        if any('time' in d['feature'] or 'hour' in d['feature'] for d in significant_drifts):
            recommendations.extend(["month_sin", "month_cos", "season_encoded"])
        variance_drifts = [d for d in significant_drifts if d['drift_type'] == 'variance_shift']
        if variance_drifts:
            recommendations.extend(["pm2_5_rolling_3d", "pm2_5_rolling_7d", "temp_rolling_3d", "aqi_rolling_7d"])
        
        return list(set(recommendations))
    
    def create_llm_prompt(self, df: pd.DataFrame, drift_results: Dict[str, Any]) -> str:
        """Use the full detailed Drift Monitoring Agent prompt template"""
        recent_data, baseline_data = self.split_time_periods(df)
        
        data_period_info = f"""
Data Period: {df.index.min()} to {df.index.max()}
Recent Period (7 days): {recent_data.index.min()} to {recent_data.index.max()}
Baseline Period (90 days): {baseline_data.index.min()} to {baseline_data.index.max()}
"""
        drift_detection_results = f"Drift Threshold: {self.drift_threshold*100}%\nTotal Features: {len(df.columns)}"
        top_drifting_features = ""
        for drift in drift_results['significant_drifts'][:5]:
            top_drifting_features += f"- {drift['feature']}: {drift['drift_type']} (Mean shift: {drift['metrics']['mean_shift']:.3f}, KS: {drift['metrics']['ks_statistic']:.3f})\n"
        
        # Insert your detailed template here
        detailed_template_path = Path("agents/drift_monitoring_agent/prompts/drift_analysis_prompt.txt")
        if not detailed_template_path.exists():
            raise FileNotFoundError(f"{detailed_template_path} not found. Please provide your detailed LLM prompt template.")
        
        with open(detailed_template_path, "r") as f:
            template = f.read()
        
        prompt = template.replace("{data_period_info}", data_period_info)\
                         .replace("{drift_detection_results}", drift_detection_results)\
                         .replace("{top_drifting_features}", top_drifting_features)
        return prompt
    
    def run_analysis(self, csv_path: str, output_path: str = None) -> Dict[str, Any]:
        """Run complete drift analysis"""
        df = self.load_data(csv_path)
        drift_results = self.detect_seasonal_shifts(df)
        feature_recommendations = self.generate_feature_recommendations(drift_results)
        llm_prompt = self.create_llm_prompt(df, drift_results)
        
        output = {
            "seasonal_shift_detected": drift_results['drift_detected'],
            "features_to_add": feature_recommendations,
            "reasoning": f"Detected {len(drift_results['significant_drifts'])} features with significant drift. "
                        f"Recommendations focus on {'pollutant' if any('pm' in str(drift_results['significant_drifts']) or 'co' in str(drift_results['significant_drifts'])) else 'weather'} "
                        f"and temporal pattern improvements.",
            "detailed_analysis": {
                "drift_metrics": drift_results['all_metrics'],
                "significant_drifts": drift_results['significant_drifts'],
                "llm_prompt": llm_prompt
            }
        }
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2, default=str)
        
        return output

if __name__ == "__main__":
    agent = DriftMonitoringAgent(drift_threshold=0.15)
    result = agent.run_analysis(
        csv_path="data/full_preprocessed_aqi_weather_data_with_all_features.csv",
        output_path="agents/drift_monitoring_agent/outputs/drift_analysis.json"
    )
    print("Drift Analysis Complete!")
    print(f"Seasonal shift detected: {result['seasonal_shift_detected']}")
    print(f"Features to add: {result['features_to_add']}")
    print(f"Reasoning: {result['reasoning']}")
