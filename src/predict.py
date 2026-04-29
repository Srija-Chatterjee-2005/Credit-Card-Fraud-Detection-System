import joblib
import pandas as pd


def predict_transaction(input_data):
    bundle = joblib.load("models/fraud_model.pkl")

    model = bundle["model"]
    features = bundle["features"]
    threshold = input_data.get("threshold", bundle["metrics"].get("threshold", 0.45))

    df = pd.DataFrame([input_data])
    df = df[features]

    probability = model.predict_proba(df)[0][1]

    if probability >= 0.75:
        risk_level = "HIGH RISK"
        decision = "BLOCK"
    elif probability >= threshold:
        risk_level = "MEDIUM RISK"
        decision = "REVIEW"
    else:
        risk_level = "LOW RISK"
        decision = "ALLOW"

    return {
        "fraud_probability": round(probability * 100, 2),
        "risk_level": risk_level,
        "decision": decision,
        "threshold_used": threshold
    }

