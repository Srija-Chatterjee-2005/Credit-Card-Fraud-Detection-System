import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    average_precision_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.data_loader import load_real_dataset
from src.feature_engineering import prepare_model_data


def calculate_fraud_cost(y_true, y_pred, fn_cost=5000, fp_cost=50):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fn * fn_cost) + (fp * fp_cost)
    return total_cost


def train_model():
    df = load_real_dataset()
    X, y = prepare_model_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train_smote, y_train_smote)

    y_proba = model.predict_proba(X_test)[:, 1]

    threshold = 0.45
    y_pred = (y_proba >= threshold).astype(int)

    fraud_cost = calculate_fraud_cost(y_test, y_pred)

    metrics = {
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "threshold": threshold,
        "fraud_cost": int(fraud_cost),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred)
    }

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": list(X.columns),
            "metrics": metrics
        },
        "models/fraud_model.pkl"
    )

    os.makedirs("outputs/reports", exist_ok=True)

    with open("outputs/reports/model_report.txt", "w") as f:
        f.write("Credit Card Fraud Detection Model Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(str(metrics))
        f.write("\n\nClassification Report:\n")
        f.write(metrics["classification_report"])

    return metrics


if __name__ == "__main__":
    print(train_model())