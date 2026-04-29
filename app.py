import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_real_dataset
from src.feature_engineering import add_dashboard_features
from src.predict import predict_transaction

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617 0%, #0f172a 40%, #111827 70%, #1e1b4b 100%);
    color: white;
}

.block-container {
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.hero {
    padding: 28px;
    border-radius: 28px;
    background: linear-gradient(135deg, #0f172a, #172554, #312e81, #7f1d1d);
    box-shadow: 0px 0px 35px rgba(59, 130, 246, 0.35);
    border: 1px solid rgba(255,255,255,0.15);
}

.hero h1 {
    font-size: 40px;
    font-weight: 900;
    color: #ffffff;
}

.hero p {
    color: #cbd5e1;
    font-size: 16px;
}

.metric-card {
    padding: 22px;
    border-radius: 24px;
    background: linear-gradient(145deg, rgba(15,23,42,0.98), rgba(30,41,59,0.95));
    border: 1px solid rgba(148,163,184,0.25);
    box-shadow: 0px 0px 20px rgba(14,165,233,0.18);
    text-align: center;
}

.metric-title {
    color: #94a3b8;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: white;
    font-size: 29px;
    font-weight: 900;
}

.metric-sub {
    color: #38bdf8;
    font-size: 12px;
}

.panel {
    padding: 22px;
    border-radius: 26px;
    background: rgba(15,23,42,0.92);
    border: 1px solid rgba(148,163,184,0.25);
    box-shadow: 0px 0px 25px rgba(168,85,247,0.18);
}

.risk-high {
    padding: 20px;
    border-radius: 22px;
    background: linear-gradient(135deg, #7f1d1d, #dc2626);
    text-align: center;
    font-size: 24px;
    font-weight: 900;
    color: white;
}

.risk-medium {
    padding: 20px;
    border-radius: 22px;
    background: linear-gradient(135deg, #92400e, #f59e0b);
    text-align: center;
    font-size: 24px;
    font-weight: 900;
    color: white;
}

.risk-low {
    padding: 20px;
    border-radius: 22px;
    background: linear-gradient(135deg, #064e3b, #10b981);
    text-align: center;
    font-size: 24px;
    font-weight: 900;
    color: white;
}

.small-text {
    color: #94a3b8;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA CHECK ----------------
if not os.path.exists("data/creditcard.csv"):
    st.error("❌ creditcard.csv not found. Please place it inside data folder.")
    st.stop()

if not os.path.exists("models/fraud_model.pkl"):
    st.error("❌ Model not found. First run: python main.py")
    st.stop()

df = load_real_dataset()
df = add_dashboard_features(df)

bundle = joblib.load("models/fraud_model.pkl")
metrics = bundle["metrics"]
features = bundle["features"]
st.write("")

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("📊 Feature Importance (Model Explainability)")

model = bundle["model"].named_steps["classifier"]

try:
    importances = model.feature_importances_
    feature_names = bundle["features"]

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(15)

    fig_fi = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#0ea5e9", "#a855f7", "#ef4444"]
    )

    fig_fi.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(fig_fi, use_container_width=True)

except Exception as e:
    st.warning(f"Feature importance not available: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🛡️ FraudShield Controls")
st.sidebar.markdown("Advanced filters for fraud-risk analysis.")

amount_min = float(df["Amount"].min())
amount_max = float(df["Amount"].max())

amount_range = st.sidebar.slider(
    "💰 Amount Range",
    min_value=amount_min,
    max_value=amount_max,
    value=(amount_min, min(5000.0, amount_max))
)

hour_range = st.sidebar.slider(
    "⏰ Hour Range",
    min_value=0,
    max_value=23,
    value=(0, 23)
)

fraud_filter = st.sidebar.multiselect(
    "🚨 Transaction Type",
    ["Normal", "Fraud"],
    default=["Normal", "Fraud"]
)

risk_amount_filter = st.sidebar.multiselect(
    "🔥 Risk Amount Level",
    list(df["Risk_Amount_Level"].dropna().unique()),
    default=list(df["Risk_Amount_Level"].dropna().unique())
)

filtered_df = df[
    (df["Amount"] >= amount_range[0]) &
    (df["Amount"] <= amount_range[1]) &
    (df["Hour"] >= hour_range[0]) &
    (df["Hour"] <= hour_range[1]) &
    (df["Fraud_Label"].isin(fraud_filter)) &
    (df["Risk_Amount_Level"].isin(risk_amount_filter))
]

threshold = st.sidebar.slider(
    "🎯 Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=float(metrics.get("threshold", 0.45)),
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.success(f"Filtered Records: {len(filtered_df):,}")

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <h1>🛡️ FraudShield AI — Fraud Operations Command Center</h1>
    <p>Real-world anonymized credit card fraud detection dashboard with ML scoring, risk intelligence, fraud alerts, threshold strategy, and downloadable model report.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- KPI CARDS ----------------
total_tx = len(filtered_df)
fraud_tx = int(filtered_df["Class"].sum())
normal_tx = total_tx - fraud_tx
fraud_rate = round((fraud_tx / total_tx) * 100, 3) if total_tx > 0 else 0
fraud_amount = round(filtered_df[filtered_df["Class"] == 1]["Amount"].sum(), 2)

c1, c2, c3, c4, c5, c6 = st.columns(6)

kpis = [
    ("Total Txns", f"{total_tx:,}", "Filtered transactions"),
    ("Normal", f"{normal_tx:,}", "Safe payments"),
    ("Fraud", f"{fraud_tx:,}", "Detected frauds"),
    ("Fraud Rate", f"{fraud_rate}%", "Imbalance view"),
    ("PR-AUC", metrics.get("pr_auc", "NA"), "Primary metric"),
    ("Fraud Amount", f"₹{fraud_amount:,.0f}", "Risk exposure"),
]

for col, item in zip([c1, c2, c3, c4, c5, c6], kpis):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{item[0]}</div>
            <div class="metric-value">{item[1]}</div>
            <div class="metric-sub">{item[2]}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# ---------------- ROW 1 CHARTS ----------------
r1c1, r1c2, r1c3 = st.columns(3)

with r1c1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📊 Fraud vs Normal Distribution")

    count_df = filtered_df["Fraud_Label"].value_counts().reset_index()
    count_df.columns = ["Type", "Count"]

    fig = px.pie(
        count_df,
        names="Type",
        values="Count",
        hole=0.55,
        color="Type",
        color_discrete_map={"Normal": "#22c55e", "Fraud": "#ef4444"}
    )
    fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with r1c2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("⏰ Fraud Activity by Hour")

    hourly = filtered_df.groupby(["Hour", "Fraud_Label"]).size().reset_index(name="Count")

    fig = px.bar(
        hourly,
        x="Hour",
        y="Count",
        color="Fraud_Label",
        barmode="group",
        color_discrete_map={"Normal": "#38bdf8", "Fraud": "#fb7185"}
    )
    fig.update_layout(
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with r1c3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("🎯 ML Performance Scorecard")

    perf_df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"],
        "Score": [
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("f1_score", 0),
            metrics.get("roc_auc", 0),
            metrics.get("pr_auc", 0)
        ]
    })

    fig = px.bar(
        perf_df,
        x="Metric",
        y="Score",
        text="Score",
        color="Metric",
        color_discrete_sequence=["#22c55e", "#ef4444", "#3b82f6", "#f59e0b", "#a855f7"]
    )
    fig.update_layout(
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font_color="white",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ---------------- ROW 2 CHARTS ----------------
r2c1, r2c2, r2c3 = st.columns(3)

with r2c1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("🔥 Amount Risk Segmentation")

    amount_risk = filtered_df.groupby(["Risk_Amount_Level", "Fraud_Label"]).size().reset_index(name="Count")

    fig = px.bar(
        amount_risk,
        x="Risk_Amount_Level",
        y="Count",
        color="Fraud_Label",
        barmode="group",
        color_discrete_map={"Normal": "#14b8a6", "Fraud": "#f97316"}
    )
    fig.update_layout(
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with r2c2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("💳 Channel Risk Simulation")

    type_risk = filtered_df.groupby(["Transaction_Type", "Fraud_Label"]).size().reset_index(name="Count")

    fig = px.bar(
        type_risk,
        x="Transaction_Type",
        y="Count",
        color="Fraud_Label",
        barmode="group",
        color_discrete_map={"Normal": "#6366f1", "Fraud": "#e11d48"}
    )
    fig.update_layout(
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with r2c3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("🧩 Confusion Matrix")

    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Fraud"],
        columns=["Pred Normal", "Pred Fraud"]
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale=["#020617", "#2563eb", "#facc15", "#ef4444"]
    )
    fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("📈 Fraud Trend Over Time")

trend = df.groupby("Hour")["Class"].mean().reset_index()

fig_trend = px.line(
    trend,
    x="Hour",
    y="Class",
    markers=True,
    line_shape="spline",
    color_discrete_sequence=["#f43f5e"]
)

fig_trend.update_layout(
    height=300,
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="white"
)

st.plotly_chart(fig_trend, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION + REPORT ----------------
left, right = st.columns([1.2, 1])

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("🧪 Live Transaction Fraud Scoring")

    st.markdown('<p class="small-text">Enter real dataset feature values. Default values are neutral for testing.</p>', unsafe_allow_html=True)

    input_data = {}

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        input_data["Time"] = st.number_input("Time", value=50000.0)
        input_data["Amount"] = st.number_input("Amount", value=250.0)

    for i in range(1, 29):
        with [p1, p2, p3, p4][i % 4]:
            input_data[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

    if st.button("🚨 Analyze Fraud Risk", use_container_width=True):
        result = predict_transaction(input_data)
        prob = result["fraud_probability"]
        risk = result["risk_level"]
        decision = result["decision"]

        if risk == "HIGH RISK":
            st.markdown(f'<div class="risk-high">🚨 {risk} | {prob}% | Decision: {decision}</div>', unsafe_allow_html=True)
        elif risk == "MEDIUM RISK":
            st.markdown(f'<div class="risk-medium">⚠️ {risk} | {prob}% | Decision: {decision}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low">✅ {risk} | {prob}% | Decision: {decision}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📄 Model Governance Report")

    report_text = f"""
Credit Card Fraud Detection System — Model Report

Model Type: XGBoost / ML Fraud Classifier
Dataset: Real-world anonymized credit card transactions
Total Transactions: {len(df):,}
Fraud Transactions: {int(df['Class'].sum()):,}
Fraud Rate: {round((df['Class'].sum()/len(df))*100, 4)}%

Performance:
Precision: {metrics.get('precision')}
Recall: {metrics.get('recall')}
F1 Score: {metrics.get('f1_score')}
ROC-AUC: {metrics.get('roc_auc')}
PR-AUC: {metrics.get('pr_auc')}
Threshold: {metrics.get('threshold', 'NA')}
Estimated Fraud Cost: {metrics.get('fraud_cost', 'NA')}

Business Interpretation:
- High recall helps catch more fraud cases.
- Precision helps reduce unnecessary false alerts.
- PR-AUC is important because fraud data is highly imbalanced.
- The system can support fraud review queues in banking and fintech workflows.
"""

    st.download_button(
        label="⬇️ Download Model Report",
        data=report_text,
        file_name="fraud_model_report.txt",
        mime="text/plain",
        use_container_width=True
    )

    st.metric("Decision Threshold", metrics.get("threshold", "NA"))
    st.metric("Estimated Fraud Cost", f"₹{metrics.get('fraud_cost', 0):,}" if isinstance(metrics.get("fraud_cost", 0), int) else "NA")
    st.metric("Model Status", "Production Simulation Ready")

    st.markdown("</div>", unsafe_allow_html=True)

if st.button("🚨 Analyze Transaction Risk"):
    result = predict_transaction(input_data)

    prob = result["fraud_probability"]
    risk = result["risk_level"]
    decision = result["decision"]

    if risk == "HIGH RISK":
        st.markdown(f'<div class="risk-high">🚨 {risk} | {prob}% | Decision: {decision}</div>', unsafe_allow_html=True)

    elif risk == "MEDIUM RISK":
        st.markdown(f'<div class="risk-medium">⚠️ {risk} | {prob}% | Decision: {decision}</div>', unsafe_allow_html=True)

    else:
        st.markdown(f'<div class="risk-low">✅ {risk} | {prob}% | Decision: {decision}</div>', unsafe_allow_html=True)

    # ✅ ADD THIS PART (do NOT replace above code)
    st.write("")

    st.markdown("### 🧠 Risk Interpretation")

    st.info("""
High fraud probability is influenced by:

• Unusual transaction patterns (PCA features)  
• High transaction amount  
• Abnormal behavioral signals  
• Time-based anomalies  

💡 In real banking systems, these patterns trigger fraud alerts and manual review queues.
""")

st.write("")

# ---------------- ALERT TABLE ----------------
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("🚨 High-Risk Fraud Alert Queue")

fraud_alerts = filtered_df[filtered_df["Class"] == 1].sort_values(by="Amount", ascending=False).head(25)

st.dataframe(
    fraud_alerts[
        [
            "Time",
            "Amount",
            "Hour",
            "Risk_Amount_Level",
            "Transaction_Type",
            "Risk_Zone",
            "Class"
        ]
    ],
    use_container_width=True,
    height=360
)

csv = fraud_alerts.to_csv(index=False)

st.download_button(
    label="⬇️ Download Fraud Alerts CSV",
    data=csv,
    file_name="fraud_alerts.csv",
    mime="text/csv"
)

st.markdown("</div>", unsafe_allow_html=True)