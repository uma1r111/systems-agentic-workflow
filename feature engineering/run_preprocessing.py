import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings("ignore")

# Load raw updated data
raw_csv = "data/karachi_weather_apr1_to_current.csv"
raw_df = pd.read_csv(raw_csv, parse_dates=['datetime'], date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
raw_df.sort_values("datetime", inplace=True)

# Load previous feature-engineered data if exists
fe_csv = "data/full_preprocessed_aqi_weather_data_with_all_features.csv"
if os.path.exists(fe_csv):
    prev_df = pd.read_csv(fe_csv, parse_dates=['datetime'], date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y %H:%M'))
    prev_df.sort_values("datetime", inplace=True)
else:
    prev_df = pd.DataFrame(columns=raw_df.columns)

# Identify new rows to process
new_df = raw_df[~raw_df['datetime'].isin(prev_df['datetime'])].copy()
if new_df.empty:
    print("✅ No new data to process. Skipping.")
    exit()

# Log transformation (apply only to new data)
skewed_cols = ["co", "pm2_5", "pm10", "precip_mm", "so2", "no2", "windspeed_kph"]
for col in skewed_cols:
    new_df[f"log_{col}"] = np.log1p(new_df[col].replace(0, np.nan)).fillna(0)  # Handle zeros

# Scaling (apply only to new data, ensure alignment)
scale_cols = [
    "temp_C", "humidity_%", "log_windspeed_kph",
    "log_pm2_5", "log_pm10", "log_precip_mm",
    "log_co", "log_no2", "log_so2", "o3"
]
scaler = StandardScaler()
scaled_values = scaler.fit_transform(new_df[scale_cols])
scaled_df = pd.DataFrame(scaled_values, columns=[f"scaled_{col}" for col in scale_cols], index=new_df.index)

# Combine new_df and scaled_df efficiently
new_df = new_df.join(scaled_df)

# Lag Features (compute on combined data to maintain consistency)
combined_df = pd.concat([prev_df[['datetime', 'aqi_us', 'scaled_log_pm10', 'scaled_log_pm2_5']], new_df[['datetime', 'aqi_us', 'scaled_log_pm10', 'scaled_log_pm2_5']]])
combined_df.sort_values("datetime", inplace=True)
combined_df.set_index("datetime", inplace=True)

# Compute lagged features
combined_df['aqi_us_lag1'] = combined_df['aqi_us'].shift(1)
combined_df['aqi_us_lag12'] = combined_df['aqi_us'].shift(12)
combined_df['aqi_us_lag24'] = combined_df['aqi_us'].shift(24)
combined_df['scaled_log_pm10_lag1'] = combined_df['scaled_log_pm10'].shift(1)
combined_df['scaled_log_pm10_lag12'] = combined_df['scaled_log_pm10'].shift(12)
combined_df['scaled_log_pm10_lag24'] = combined_df['scaled_log_pm10'].shift(24)
combined_df['scaled_log_pm2_5_lag1'] = combined_df['scaled_log_pm2_5'].shift(1)
combined_df['scaled_log_pm2_5_lag12'] = combined_df['scaled_log_pm2_5'].shift(12)
combined_df['scaled_log_pm2_5_lag24'] = combined_df['scaled_log_pm2_5'].shift(24)

# Reset index to make datetime a column again
combined_df.reset_index(inplace=True)

# Merge lags back to new_df
new_df = new_df.reset_index().merge(combined_df[['datetime', 'aqi_us_lag1', 'aqi_us_lag12', 'aqi_us_lag24',
                'scaled_log_pm10_lag1', 'scaled_log_pm10_lag12', 'scaled_log_pm10_lag24',
                'scaled_log_pm2_5_lag1', 'scaled_log_pm2_5_lag12', 'scaled_log_pm2_5_lag24']], 
                on='datetime', how='left')

# Time-based Features
new_df.set_index('datetime', inplace=True)
new_df['hour'] = new_df.index.hour
new_df['day_of_week'] = new_df.index.dayofweek
new_df['is_weekend'] = new_df['day_of_week'].isin([5, 6]).astype(int)
new_df['hour_sin'] = np.sin(new_df['hour'] * 2 * np.pi / 24)
new_df['hour_cos'] = np.cos(new_df['hour'] * 2 * np.pi / 24)

# Interaction Features
new_df['log_pm2_5_scaled_log_windspeed_kph'] = new_df['log_pm2_5'] * new_df['scaled_log_windspeed_kph']
new_df['scaled_temp_C_scaled_o3'] = new_df['scaled_temp_C'] * new_df['scaled_o3']
new_df['scaled_temp_C_scaled_log_windspeed_kph'] = new_df['scaled_temp_C'] * new_df['scaled_log_windspeed_kph']

# Combine and save final dataset
final_df = pd.concat([prev_df, new_df.reset_index()], ignore_index=True)
final_df.drop_duplicates(subset="datetime", keep="last", inplace=True)
final_df.sort_values("datetime", inplace=True)
final_df['datetime'] = final_df['datetime'].dt.strftime('%d/%m/%Y %H:%M')  # Explicitly format before saving
final_df.drop(columns=['index'], errors='ignore', inplace=True)  # Drop the extra index column if it exists

final_path = "data/full_preprocessed_aqi_weather_data_with_all_features.csv"
final_df.to_csv(final_path, index=False)
print(f"✅ Saved: {final_path} with shape {final_df.shape}")