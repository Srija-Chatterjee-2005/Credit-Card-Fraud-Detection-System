import pandas as pd
import os

def load_real_dataset(path="data/creditcard.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "❌ creditcard.csv not found inside data/ folder"
        )

    df = pd.read_csv(path)

    if "Class" not in df.columns:
        raise ValueError("❌ Dataset must contain 'Class' column")

    return df