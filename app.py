"""Customer Churn Prediction — Streamlit app."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
META_PATH = ROOT / "models" / "model_meta.joblib"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return pipeline, meta


@st.cache_data
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    return df


def risk_label(proba: float) -> tuple[str, str]:
    if proba >= 0.65:
        return "High risk", "🔴"
    if proba >= 0.40:
        return "Medium risk", "🟡"
    return "Low risk", "🟢"


def main() -> None:
    pipeline, meta = load_artifacts()
    df = load_dataset()

    st.title("Customer Churn Predictor")
    st.caption(
        "Predict which telecom customers are likely to leave — "
        "built for free hosting on Streamlit Community Cloud / Hugging Face Spaces."
    )

    # --- Overview metrics ---
    churn_count = int((df["Churn"] == "Yes").sum())
    stay_count = len(df) - churn_count
    churn_rate = churn_count / len(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churned", f"{churn_count:,}")
    c3.metric("Retained", f"{stay_count:,}")
    c4.metric("Churn rate", f"{churn_rate:.1%}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Model accuracy", f"{meta['accuracy']:.1%}")
    m2.metric("F1 (churn)", f"{meta['f1']:.1%}")
    m3.metric("ROC AUC", f"{meta['roc_auc']:.1%}")

    st.divider()

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Predict for a customer")
        with st.form("churn_form"):
            g1, g2 = st.columns(2)
            with g1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                phone = st.selectbox("Phone service", ["Yes", "No"])
                multi = st.selectbox(
                    "Multiple lines",
                    ["No", "Yes", "No phone service"],
                )
                internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
                online_sec = st.selectbox(
                    "Online security",
                    ["No", "Yes", "No internet service"],
                )
                online_bak = st.selectbox(
                    "Online backup",
                    ["No", "Yes", "No internet service"],
                )

            with g2:
                device = st.selectbox(
                    "Device protection",
                    ["No", "Yes", "No internet service"],
                )
                tech = st.selectbox(
                    "Tech support",
                    ["No", "Yes", "No internet service"],
                )
                stream_tv = st.selectbox(
                    "Streaming TV",
                    ["No", "Yes", "No internet service"],
                )
                stream_mov = st.selectbox(
                    "Streaming movies",
                    ["No", "Yes", "No internet service"],
                )
                contract = st.selectbox(
                    "Contract",
                    ["Month-to-month", "One year", "Two year"],
                )
                paperless = st.selectbox("Paperless billing", ["Yes", "No"])
                payment = st.selectbox(
                    "Payment method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                )
                monthly = st.number_input(
                    "Monthly charges ($)",
                    min_value=0.0,
                    max_value=200.0,
                    value=70.0,
                    step=0.5,
                )
                total = st.number_input(
                    "Total charges ($)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=float(max(monthly * tenure, monthly)),
                    step=1.0,
                )

            submitted = st.form_submit_button("Predict churn risk", type="primary", use_container_width=True)

        if submitted:
            row = pd.DataFrame(
                [
                    {
                        "gender": gender,
                        "SeniorCitizen": senior,
                        "Partner": partner,
                        "Dependents": dependents,
                        "tenure": tenure,
                        "PhoneService": phone,
                        "MultipleLines": multi,
                        "InternetService": internet,
                        "OnlineSecurity": online_sec,
                        "OnlineBackup": online_bak,
                        "DeviceProtection": device,
                        "TechSupport": tech,
                        "StreamingTV": stream_tv,
                        "StreamingMovies": stream_mov,
                        "Contract": contract,
                        "PaperlessBilling": paperless,
                        "PaymentMethod": payment,
                        "MonthlyCharges": monthly,
                        "TotalCharges": total,
                    }
                ]
            )
            proba = float(pipeline.predict_proba(row)[0, 1])
            label, icon = risk_label(proba)

            st.markdown(f"### {icon} {label}")
            st.progress(min(proba, 1.0))
            st.write(f"**Churn probability:** {proba:.1%}")
            if proba >= 0.65:
                st.info(
                    "Suggested action: reach out with a retention offer "
                    "(discount, plan upgrade, or dedicated support)."
                )
            elif proba >= 0.40:
                st.info(
                    "Suggested action: check satisfaction and highlight unused benefits."
                )
            else:
                st.success("Customer looks stable — keep standard engagement.")

    with right:
        st.subheader("Top feature importances")
        st.caption("What the model weighs most when scoring churn risk.")
        pairs = meta["feature_importances"]
        names = [n.replace("cat__", "").replace("num__", "") for n, _ in pairs]
        values = [v for _, v in pairs]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(names[::-1], values[::-1], color="#2a6f97")
        ax.set_xlabel("Importance")
        ax.set_title("Random Forest — top features")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Churn by contract type")
        contract_churn = (
            df.assign(is_churn=(df["Churn"] == "Yes").astype(int))
            .groupby("Contract")["is_churn"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(contract_churn)

    with st.expander("About this project"):
        st.markdown(
            """
            - **Dataset:** IBM Telco Customer Churn (~7k customers)
            - **Model:** Random Forest with one-hot encoding (scikit-learn pipeline)
            - **Host free on:** [Streamlit Community Cloud](https://share.streamlit.io)
              or [Hugging Face Spaces](https://huggingface.co/spaces) (Streamlit SDK)
            - Retrain locally with: `python train_model.py`
            """
        )


if __name__ == "__main__":
    main()
