import pandas as pd
import numpy as np


def add_dashboard_features(df):
    df = df.copy()

    df["Hour"] = ((df["Time"] // 3600) % 24).astype(int)

    df["Risk_Amount_Level"] = pd.cut(
        df["Amount"],
        bins=[-1, 50, 250, 1000, 5000, float("inf")],
        labels=["Very Low", "Low", "Medium", "High", "Extreme"]
    )

    np.random.seed(42)

    df["Transaction_Type"] = np.random.choice(
        ["POS", "Online", "ATM", "Wallet", "International"],
        size=len(df),
        p=[0.35, 0.30, 0.12, 0.15, 0.08]
    )

    df["Risk_Zone"] = np.where(
        df["Class"] == 1,
        np.random.choice(["High Risk", "Critical Risk"], size=len(df)),
        np.random.choice(["Low Risk", "Moderate Risk"], size=len(df))
    )

    df["Fraud_Label"] = df["Class"].map({0: "Normal", 1: "Fraud"})

    return df


def prepare_model_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y