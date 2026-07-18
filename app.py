"""Customer Churn Predictor — interview-ready Streamlit app."""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.batch import (
    REQUIRED_INPUT_COLS,
    score_customers,
    validate_upload,
    verification_summary,
)
from src.config import (
    COMPARISON_PATH,
    CONFUSION_PATH,
    DATA_PATH,
    FEATURE_COLS,
    META_PATH,
    METRICS_PATH,
    PIPELINE_PATH,
    ROC_PATH,
    ROOT,
)
from src.data import clean_data, load_raw
from src.explain import insight_bullets, risk_band
from src.features import engineer_features

SAMPLE_UPLOAD_PATH = ROOT / "data" / "sample_upload.csv"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)


@st.cache_resource
def load_pipeline_and_meta():
    pipeline = joblib.load(PIPELINE_PATH)
    meta = joblib.load(META_PATH)
    return pipeline, meta


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


def render_overview(df: pd.DataFrame, meta: dict, reports: dict) -> None:
    # clean_data encodes Churn as 0/1
    churn_count = int(df["Churn"].sum())
    stay_count = len(df) - churn_count
    churn_rate = float(df["Churn"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churned", f"{churn_count:,}")
    c3.metric("Retained", f"{stay_count:,}")
    c4.metric("Churn rate", f"{churn_rate:.1%}")

    tuned = meta.get("metrics_at_tuned") or meta
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best model", meta.get("best_model", "n/a").replace("_", " ").title())
    m2.metric("ROC AUC", f"{meta.get('metrics_at_0_5', {}).get('roc_auc', meta.get('roc_auc', 0)):.1%}")
    m3.metric("Tuned threshold", f"{meta.get('threshold', 0.5):.2f}")
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
        pairs = meta.get("feature_importances", [])
        if pairs:
            names = [p[0] if isinstance(p, (list, tuple)) else p["feature"] for p in pairs]
            values = [p[1] if isinstance(p, (list, tuple)) else p["importance"] for p in pairs]
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.barh(names[::-1], values[::-1], color="#2a6f97")
            ax.set_xlabel("Importance / |coefficient|")
            ax.set_title(f"{meta.get('best_model', 'model')} — top features")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.info(
        "Selection rule: highest ROC AUC on a stratified holdout; "
        "decision threshold tuned so missed churners cost more than wasted outreach."
    )


def render_predict(pipeline, meta: dict) -> None:
    threshold = float(meta.get("threshold", 0.5))
    st.caption(f"Business-tuned decision threshold = **{threshold:.2f}**")

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
            monthly = st.number_input(
                "Monthly charges ($)", 0.0, 200.0, 70.0, 0.5
            )
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
                "Churn": 0,
            }
        ]
    )
    row = engineer_features(row)
    X = row[FEATURE_COLS]
    proba = float(pipeline.predict_proba(X)[0, 1])
    label, color = risk_band(proba, threshold)
    decision = "Likely to churn" if proba >= threshold else "Likely to stay"

    st.markdown(f"### {label}")
    st.progress(min(max(proba, 0.0), 1.0))
    st.write(f"**Churn probability:** {proba:.1%}")
    st.write(f"**Decision @ {threshold:.2f}:** {decision}")

    if proba >= threshold + 0.15:
        st.warning(
            "Suggested action: prioritized retention offer "
            "(discount, plan change, or dedicated support)."
        )
    elif proba >= threshold:
        st.info("Suggested action: check satisfaction and highlight unused benefits.")
    else:
        st.success("Customer looks stable — keep standard engagement.")


def render_model_lab(reports: dict, meta: dict) -> None:
    if "comparison" in reports:
        st.subheader("Model comparison (holdout, threshold=0.5)")
        st.dataframe(
            reports["comparison"].style.format(
                {
                    "accuracy": "{:.3f}",
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1": "{:.3f}",
                    "roc_auc": "{:.3f}",
                    "threshold": "{:.2f}",
                }
            ),
            use_container_width=True,
        )
        st.caption(
            f"Winner: **{meta.get('best_model', 'n/a')}** by ROC AUC. "
            "Imbalance handled with stratified split + class weights (where supported)."
        )
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
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
            ax.set_title(f"threshold={cm['threshold']:.2f}")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    metrics = reports.get("metrics", {})
    if metrics:
        with st.expander("Methodology notes"):
            st.write(metrics.get("selection_rule", ""))
            st.write(metrics.get("imbalance_strategy", ""))
            st.json(metrics.get("cost_matrix", {}))


def render_upload(pipeline, meta: dict) -> None:
    """Upload a Telco-style CSV → score each row with the trained model."""
    threshold = float(meta.get("threshold", 0.5))
    st.subheader("Batch demo — upload CSV")
    st.markdown(
        """
        **Where does the model get its knowledge?**  
        It was trained once on the bundled IBM Telco dataset
        (`data/Telco-Customer-Churn.csv`) and saved in `models/churn_pipeline.joblib`.
        Your upload is **not** used to retrain — it is only scored.

        **How verification works:**  
        If your CSV includes a `Churn` column (`Yes`/`No`), we compare the model's
        prediction to that label and show accuracy. Without `Churn`, you still get
        risk scores for every row.
        """
    )

    st.caption(f"Decision threshold = **{threshold:.2f}** (from training cost tuning)")

    if SAMPLE_UPLOAD_PATH.exists():
        st.download_button(
            "Download sample CSV (20 rows)",
            data=SAMPLE_UPLOAD_PATH.read_bytes(),
            file_name="sample_upload.csv",
            mime="text/csv",
        )

    with st.expander("Required columns"):
        st.code(", ".join(REQUIRED_INPUT_COLS))
        st.write("Optional: `customerID`, `Churn` (for verification)")

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read CSV: {exc}")
        return

    missing = validate_upload(raw)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    st.write(f"Uploaded **{len(raw):,}** rows. Preview:")
    st.dataframe(raw.head(10), use_container_width=True)

    if st.button("Score uploaded customers", type="primary"):
        with st.spinner("Scoring..."):
            scored = score_customers(pipeline, raw, threshold)
            summary = verification_summary(scored)

        pred_yes = int((scored["predicted_churn"] == "Yes").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows scored", f"{len(scored):,}")
        c2.metric("Predicted churn", f"{pred_yes:,}")
        c3.metric("Predicted churn rate", f"{pred_yes / len(scored):.1%}")

        if summary:
            st.success(
                f"Verification vs `Churn` labels: "
                f"**{summary['correct']}/{summary['rows']}** correct "
                f"({summary['accuracy']:.1%} accuracy)"
            )
            v1, v2 = st.columns(2)
            v1.metric("Actual churn rate", f"{summary['actual_churn_rate']:.1%}")
            v2.metric("Predicted churn rate", f"{summary['predicted_churn_rate']:.1%}")
        else:
            st.info(
                "No `Churn` column in file — showing predictions only "
                "(nothing to verify against)."
            )

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


def render_insights() -> None:
    st.subheader("Insights you can cite in interviews")
    for i, bullet in enumerate(insight_bullets(), start=1):
        st.markdown(f"{i}. {bullet}")
    st.divider()
    st.markdown(
        """
        **How to walk through this project (2 minutes):**
        1. Business problem → reduce revenue loss from churn  
        2. EDA → contract, tenure, payment method dominate  
        3. Features → `tenure_bucket`, `avg_monthly_spend`, `service_count`  
        4. Models → LR / RF / GBM compared on ROC AUC  
        5. Threshold → cost matrix (FN > FP), not default 0.5  
        6. Demo → live risk score in this app  
        """
    )


def main() -> None:
    pipeline, meta = load_pipeline_and_meta()
    reports = load_reports()
    df = load_dataset()

    st.title("Customer Churn Predictor")
    st.caption(
        "Predict telecom customer churn from service and billing data — "
        "score one customer or upload a CSV for batch predictions."
    )

    tab_overview, tab_predict, tab_upload, tab_lab, tab_insights = st.tabs(
        ["Overview", "Predict", "Upload CSV", "Model Lab", "Insights"]
    )
    with tab_overview:
        render_overview(df, meta, reports)
    with tab_predict:
        render_predict(pipeline, meta)
    with tab_upload:
        render_upload(pipeline, meta)
    with tab_lab:
        render_model_lab(reports, meta)
    with tab_insights:
        render_insights()


if __name__ == "__main__":
    main()
