"""Streamlit UI — thin client over the FastAPI churn service."""

from __future__ import annotations

import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.batch import REQUIRED_INPUT_COLS
from src.config import (
    COMPARISON_PATH,
    CONFUSION_PATH,
    DATA_PATH,
    METRICS_PATH,
    ROC_PATH,
    ROOT,
)
from src.data import clean_data, load_raw
from src.explain import insight_bullets

SAMPLE_UPLOAD_PATH = ROOT / "data" / "sample_upload.csv"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)


@st.cache_data
def load_reports() -> dict:
    reports = {}
    if METRICS_PATH.exists():
        reports["metrics"] = json.loads(METRICS_PATH.read_text())
    if COMPARISON_PATH.exists():
        reports["comparison"] = pd.read_csv(COMPARISON_PATH)
    if ROC_PATH.exists():
        reports["roc"] = json.loads(ROC_PATH.read_text())
    if CONFUSION_PATH.exists():
        reports["confusion"] = json.loads(CONFUSION_PATH.read_text())
    return reports


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return clean_data(load_raw(DATA_PATH))


def api_get(path: str, timeout: float = 10.0):
    return requests.get(f"{API_URL}{path}", timeout=timeout)


def api_post_json(path: str, payload: dict, timeout: float = 30.0):
    return requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)


def api_post_file(path: str, file_bytes: bytes, filename: str, timeout: float = 60.0):
    return requests.post(
        f"{API_URL}{path}",
        files={"file": (filename, file_bytes, "text/csv")},
        timeout=timeout,
    )


def render_api_status() -> dict | None:
    try:
        r = api_get("/health")
        r.raise_for_status()
        health = r.json()
        if health.get("model_loaded"):
            st.sidebar.success(f"API connected · {API_URL}")
        else:
            st.sidebar.warning("API up but model not loaded")
        return health
    except requests.RequestException as exc:
        st.sidebar.error(f"API unreachable ({API_URL}): {exc}")
        return None


def render_overview(df: pd.DataFrame, reports: dict, health: dict | None) -> None:
    churn_count = int(df["Churn"].sum())
    stay_count = len(df) - churn_count
    churn_rate = float(df["Churn"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churned", f"{churn_count:,}")
    c3.metric("Retained", f"{stay_count:,}")
    c4.metric("Churn rate", f"{churn_rate:.1%}")

    metrics = reports.get("metrics", {})
    tuned = metrics.get("metrics_at_tuned", {})
    m1, m2, m3, m4 = st.columns(4)
    best = (health or {}).get("best_model") or metrics.get("best_model", "n/a")
    thr = (health or {}).get("threshold") or metrics.get("threshold", 0.5)
    m1.metric("Best model", str(best).replace("_", " ").title())
    m2.metric(
        "ROC AUC",
        f"{metrics.get('metrics_at_0_5', {}).get('roc_auc', 0):.1%}",
    )
    m3.metric("Tuned threshold", f"{float(thr):.2f}")
    m4.metric("Recall @ threshold", f"{tuned.get('recall', 0):.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Churn by contract")
        contract_churn = (
            df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
        )
        st.bar_chart(contract_churn)

    with right:
        st.subheader("Top drivers")
        pairs = metrics.get("feature_importances", [])
        if pairs:
            names = [p["feature"] if isinstance(p, dict) else p[0] for p in pairs]
            values = [p["importance"] if isinstance(p, dict) else p[1] for p in pairs]
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.barh(names[::-1], values[::-1], color="#2a6f97")
            ax.set_xlabel("Importance")
            ax.set_title(f"{best} — top features")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.info(
        "UI calls the FastAPI service for live scoring. "
        "Charts below use bundled training reports and dataset."
    )


def render_predict(health: dict | None) -> None:
    if not health or not health.get("model_loaded"):
        st.warning("Start the API (`uvicorn api.main:app --reload`) then refresh.")
        return

    threshold = float(health.get("threshold") or 0.5)
    st.caption(f"Scoring via `{API_URL}/predict` · threshold = **{threshold:.2f}**")

    with st.form("churn_form"):
        g1, g2 = st.columns(2)
        with g1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox(
                "Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No"
            )
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            online_sec = st.selectbox(
                "Online security", ["No", "Yes", "No internet service"]
            )
            online_bak = st.selectbox(
                "Online backup", ["No", "Yes", "No internet service"]
            )
        with g2:
            device = st.selectbox(
                "Device protection", ["No", "Yes", "No internet service"]
            )
            tech = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            stream_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            stream_mov = st.selectbox(
                "Streaming movies", ["No", "Yes", "No internet service"]
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
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
            monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, 0.5)
            total = st.number_input(
                "Total charges ($)",
                0.0,
                10000.0,
                float(max(monthly * max(tenure, 1), monthly)),
                1.0,
            )

        submitted = st.form_submit_button(
            "Predict churn risk", type="primary", use_container_width=True
        )

    if not submitted:
        return

    payload = {
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

    try:
        r = api_post_json("/predict", payload)
        if r.status_code >= 400:
            st.error(r.json().get("detail", r.text))
            return
        out = r.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return

    proba = float(out["churn_probability"])
    st.markdown(f"### {out['risk_band']}")
    st.progress(min(max(proba, 0.0), 1.0))
    st.write(f"**Churn probability:** {proba:.1%}")
    st.write(f"**Decision:** {out['decision']}")
    st.json(out)


def render_upload(health: dict | None) -> None:
    if not health or not health.get("model_loaded"):
        st.warning("API must be running for batch scoring.")
        return

    st.subheader("Batch demo — upload CSV")
    st.markdown(
        f"""
        Uploads go to `{API_URL}/predict/csv`. The API validates columns, scores rows,
        writes an audit row to SQLite, and returns verification stats when `Churn` is present.
        """
    )

    if SAMPLE_UPLOAD_PATH.exists():
        st.download_button(
            "Download sample CSV (20 rows)",
            data=SAMPLE_UPLOAD_PATH.read_bytes(),
            file_name="sample_upload.csv",
            mime="text/csv",
        )

    with st.expander("Required columns"):
        st.code(", ".join(REQUIRED_INPUT_COLS))

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is None:
        return

    raw = uploaded.getvalue()
    preview = pd.read_csv(io.BytesIO(raw))
    st.write(f"Uploaded **{len(preview):,}** rows. Preview:")
    st.dataframe(preview.head(10), use_container_width=True)

    if st.button("Score uploaded customers", type="primary"):
        with st.spinner("Calling API..."):
            try:
                r = api_post_file("/predict/csv", raw, uploaded.name or "upload.csv")
                if r.status_code >= 400:
                    detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
                    st.error(detail)
                    return
                out = r.json()
            except requests.RequestException as exc:
                st.error(f"API error: {exc}")
                return

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows scored", f"{out['count']:,}")
        c2.metric("Predicted churn", f"{out['predicted_churn_count']:,}")
        c3.metric("Predicted churn rate", f"{out['predicted_churn_rate']:.1%}")

        if out.get("verification"):
            v = out["verification"]
            st.success(
                f"Verification: **{v['correct']}/{v['rows']}** correct "
                f"({v['accuracy']:.1%} accuracy)"
            )
        else:
            st.info("No `Churn` column — predictions only.")

        scored = pd.DataFrame(out["results"])
        show_cols = [
            c
            for c in [
                "customerID",
                "Contract",
                "tenure",
                "MonthlyCharges",
                "churn_probability",
                "predicted_churn",
                "risk_band",
                "actual_churn",
                "correct",
            ]
            if c in scored.columns
        ]
        st.dataframe(
            scored[show_cols].sort_values("churn_probability", ascending=False),
            use_container_width=True,
        )
        st.download_button(
            "Download scored CSV",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="churn_scored.csv",
            mime="text/csv",
        )


def render_history(health: dict | None) -> None:
    if not health:
        st.warning("API offline — history unavailable.")
        return
    try:
        r = api_get("/predictions?limit=50")
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as exc:
        st.error(f"Could not load history: {exc}")
        return

    st.subheader("Recent API predictions (SQLite audit log)")
    st.caption(f"{data['count']} latest records from `{API_URL}/predictions`")
    if not data["items"]:
        st.info("No predictions logged yet. Score a customer first.")
        return
    st.dataframe(pd.DataFrame(data["items"]), use_container_width=True)


def render_model_lab(reports: dict) -> None:
    if "comparison" in reports:
        st.subheader("Model comparison (holdout, threshold=0.5)")
        st.dataframe(reports["comparison"], use_container_width=True)
    else:
        st.warning("Run `python train_model.py` to generate comparison reports.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROC curve (best model)")
        roc = reports.get("roc")
        if roc:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(roc["fpr"], roc["tpr"], color="#2a6f97", label=f"AUC={roc['auc']:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(loc="lower right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with c2:
        st.subheader("Confusion matrix @ tuned threshold")
        cm = reports.get("confusion")
        if cm:
            matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
            fig, ax = plt.subplots(figsize=(4.5, 4))
            im = ax.imshow(matrix, cmap="Blues")
            ax.set_xticks([0, 1], labels=["Pred Stay", "Pred Churn"])
            ax.set_yticks([0, 1], labels=["Actual Stay", "Actual Churn"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
            ax.set_title(f"threshold={cm['threshold']:.2f}")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


def render_insights() -> None:
    st.subheader("Product insights")
    for i, bullet in enumerate(insight_bullets(), start=1):
        st.markdown(f"{i}. {bullet}")
    st.divider()
    st.markdown(
        """
        **Engineering walkthrough:**
        1. Client (Streamlit) → HTTP → FastAPI  
        2. Pydantic validates the payload  
        3. PredictionService scores with the saved pipeline  
        4. Result persisted to SQLite audit log  
        5. OpenAPI docs at `/docs`  
        """
    )


def main() -> None:
    reports = load_reports()
    df = load_dataset()
    health = render_api_status()

    st.title("Customer Churn Predictor")
    st.caption(
        "Predict telecom customer churn from service and billing data — "
        "score one customer or upload a CSV for batch predictions."
    )

    tabs = st.tabs(
        ["Overview", "Predict", "Upload CSV", "History", "Model Lab", "Insights"]
    )
    with tabs[0]:
        render_overview(df, reports, health)
    with tabs[1]:
        render_predict(health)
    with tabs[2]:
        render_upload(health)
    with tabs[3]:
        render_history(health)
    with tabs[4]:
        render_model_lab(reports)
    with tabs[5]:
        render_insights()


if __name__ == "__main__":
    main()
