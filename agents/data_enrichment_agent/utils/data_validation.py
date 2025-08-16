# utils/data_validation.py
from typing import Any, Optional
from io import StringIO
import json
import logging

import pandas as pd
import numpy as np

NUMERIC_ABS_LIMIT = 1e8  # values larger than this considered unrealistic
DATETIME_PARSE_RATIO = 0.6  # fraction required when inferring datetime column


class DataValidator:
    """
    Schema-agnostic validator/cleaner for enrichment agent.
    Usage:
        dv = DataValidator(logger=my_logger)   # optional logger
        df = dv.validate_and_clean(raw_data, feature_name="wind_direction")
        if df is None:
            # validation failed
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("DataValidator")

    # -------------------------
    # Public API
    # -------------------------
    def validate_and_clean(self, raw_data: Any, feature_name: str = "") -> Optional[pd.DataFrame]:
        """
        Convert raw_data into a cleaned DataFrame if possible.
        Returns DataFrame on success, None on failure.
        """
        try:
            df = self._to_dataframe(raw_data)
            if df is None:
                self.logger.warning(f"[{feature_name}] unable to convert raw data to DataFrame")
                return None

            if df.empty:
                self.logger.warning(f"[{feature_name}] dataframe is empty after conversion")
                return None

            # Drop columns that are entirely null
            df = df.dropna(axis=1, how="all")

            # Drop duplicate rows
            df = df.drop_duplicates().reset_index(drop=True)

            # Try detect and parse datetime column (best-effort)
            dt_col = self._detect_datetime_column(df)
            if dt_col is not None:
                df = self._parse_and_sort_datetime(df, dt_col, feature_name)

            # Validate numeric sanity
            if not self._numeric_sanity_check(df, feature_name):
                return None

            # Final basic check: not empty
            if df.empty:
                self.logger.warning(f"[{feature_name}] dataframe empty after cleaning steps")
                return None

            return df

        except Exception as e:
            self.logger.exception(f"[{feature_name}] unexpected error in validation: {e}")
            return None

    # -------------------------
    # Internal helpers
    # -------------------------
    def _to_dataframe(self, raw: Any) -> Optional[pd.DataFrame]:
        """Accept DataFrame, list/dict, JSON string, or CSV string. Return DataFrame or None."""
        # If already a DataFrame
        if isinstance(raw, pd.DataFrame):
            return raw.copy()

        # If it's a requests.Response-like object
        if hasattr(raw, "json") and callable(raw.json):
            try:
                parsed = raw.json()
                return self._to_dataframe(parsed)
            except Exception:
                pass
        if hasattr(raw, "text"):
            text = raw.text
            # fall through to string handling

        # If raw is bytes/str: try JSON then CSV
        if isinstance(raw, (bytes, str)):
            s = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            s = s.strip()
            # JSON-like?
            if (s.startswith("{") or s.startswith("[")):
                try:
                    parsed = json.loads(s)
                    return self._to_dataframe(parsed)
                except Exception:
                    pass
            # CSV-like?
            try:
                return pd.read_csv(StringIO(s))
            except Exception:
                pass

        # If it's a list or dict (parsed JSON)
        if isinstance(raw, list):
            try:
                return pd.DataFrame(raw)
            except Exception:
                try:
                    return pd.json_normalize(raw)
                except Exception:
                    return None

        if isinstance(raw, dict):
            # If top-level contains an array under common keys, use it
            for key in ("data", "results", "observations", "records", "values", "measurements", "list"):
                if key in raw and isinstance(raw[key], (list, dict)):
                    try:
                        return self._to_dataframe(raw[key])
                    except Exception:
                        pass
            # Fallback to normalize dict
            try:
                return pd.json_normalize(raw)
            except Exception:
                return None

        # Unknown type
        return None

    def _detect_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Return column name likely to contain datetime information or None."""
        # Candidate names
        candidate_names = [c for c in df.columns if any(k in c.lower() for k in ("datetime", "date", "time", "timestamp"))]
        if candidate_names:
            return candidate_names[0]

        # Try inferring: pick the column that best parses as datetime
        best_col = None
        best_ratio = 0.0
        n = len(df)
        for col in df.columns:
            # Try only on string/object-like columns
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col  # already datetime dtype
            try:
                parsed = pd.to_datetime(df[col], errors="coerce", utc=False)
            except Exception:
                continue
            non_na = parsed.notna().sum()
            ratio = non_na / max(1, n)
            if ratio > best_ratio:
                best_ratio = ratio
                best_col = col
        if best_ratio >= DATETIME_PARSE_RATIO:
            return best_col
        return None

    def _parse_and_sort_datetime(self, df: pd.DataFrame, col: str, feature_name: str) -> pd.DataFrame:
        """Parse a datetime column to pandas datetime, drop rows with NaT, sort by it, name it 'datetime'."""
        parsed = pd.to_datetime(df[col], errors="coerce", utc=False)
        valid_count = parsed.notna().sum()
        total = len(parsed)
        if valid_count == 0:
            self.logger.warning(f"[{feature_name}] detected datetime column '{col}' but none values parsed")
            return df  # keep original if parse fails

        df = df.assign(_parsed_datetime=parsed)
        # Drop rows where datetime couldn't parse
        df = df[df["_parsed_datetime"].notna()].copy()
        df = df.sort_values("_parsed_datetime").reset_index(drop=True)
        # Canonicalize: place parsed datetime in column named 'datetime' (overwrite if exists)
        df["datetime"] = df["_parsed_datetime"]
        df = df.drop(columns=["_parsed_datetime"])
        return df

    def _numeric_sanity_check(self, df: pd.DataFrame, feature_name: str) -> bool:
        """Check for infinities / nan / absurd magnitudes in numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            # No numeric columns isn't an automatic fail for generic validator
            return True

        # Check for infs or NaNs
        for col in numeric_cols:
            series = df[col]
            if np.isinf(series).any():
                self.logger.warning(f"[{feature_name}] numeric column '{col}' contains infinite values")
                return False
            # If any absolute value > limit => suspect bad API
            if (series.abs() > NUMERIC_ABS_LIMIT).any():
                self.logger.warning(f"[{feature_name}] numeric column '{col}' contains extreme values (>|{NUMERIC_ABS_LIMIT}|)")
                return False

        return True
