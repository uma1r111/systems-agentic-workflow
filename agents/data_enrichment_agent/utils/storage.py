# utils/storage.py
import os
import pandas as pd
import json

class DataStorage:
    def __init__(self, base_dir: str = "data/enrichment"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_csv(self, df: pd.DataFrame, filename: str):
        path = os.path.join(self.base_dir, filename)
        df.to_csv(path, index=False)
        return path

    def load_csv(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.base_dir, filename)
        return pd.read_csv(path)

    def save_json(self, data: dict, filename: str):
        path = os.path.join(self.base_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def load_json(self, filename: str) -> dict:
        path = os.path.join(self.base_dir, filename)
        with open(path, "r") as f:
            return json.load(f)
