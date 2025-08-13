import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from scipy import stats
from typing import Dict, List, Any
import os
from pathlib import Path
import google.generativeai as genai

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
            df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)  # <-- add dayfirst=True
            df.set_index('datetime', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], dayfirst=True)  # <-- add dayfirst=True
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
    
    def create_llm_prompt(self, df: pd.DataFrame, drift_results: Dict[str, Any]) -> str:
        """Generate a prompt for the LLM with all drift metrics and details"""
        recent_data, baseline_data = self.split_time_periods(df)

        data_period_info = f"""
    Data Period: {df.index.min()} to {df.index.max()}
    Recent Period (7 days): {recent_data.index.min()} to {recent_data.index.max()}
    Baseline Period (90 days): {baseline_data.index.min()} to {baseline_data.index.max()}
    """

        drift_summary = []
        for drift in drift_results['significant_drifts']:
            drift_summary.append(
                f"- Feature: {drift['feature']}, Drift Type: {drift['drift_type']}, "
                f"Mean Shift: {drift['metrics']['mean_shift']:.3f}, "
                f"Variance Shift: {drift['metrics']['variance_shift']:.3f}, "
                f"KS Statistic: {drift['metrics']['ks_statistic']:.3f}"
            )

        drift_summary_text = "\n".join(drift_summary) if drift_summary else "No significant drift detected."

        # Load detailed prompt template
        template_path = Path("agents/drift_monitoring_agent/prompts/drift_analysis_prompt.txt")
        if not template_path.exists():
            raise FileNotFoundError(f"{template_path} not found.")

        with open(template_path, "r") as f:
            template = f.read()

        prompt = template.replace("{data_period_info}", data_period_info)\
                        .replace("{drift_summary}", drift_summary_text)\
                        .replace("{all_features}", ", ".join(df.columns))\
                        .replace("{drift_threshold}", f"{self.drift_threshold*100:.1f}%")

        return prompt


    def run_analysis(self, csv_path: str, output_path: str = None) -> Dict[str, Any]:
        """Run full drift analysis and let LLM suggest features and reasoning"""
        df = self.load_data(csv_path)
        drift_results = self.detect_seasonal_shifts(df)
        llm_prompt = self.create_llm_prompt(df, drift_results)

        # Call Gemini / LLM to get recommended features and reasoning
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content(
            llm_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=500,
                response_mime_type="application/json"
            )
        )

        llm_output = json.loads(response.text)

        output = {
            "seasonal_shift_detected": drift_results['drift_detected'],
            "features_to_add": llm_output.get("features_to_add", []),
            "reasoning": llm_output.get("reasoning", ""),
            "detailed_analysis": {
                "drift_metrics": drift_results['all_metrics'],
                "significant_drifts": drift_results['significant_drifts'],
                "llm_prompt": llm_prompt
            }
        }

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
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
