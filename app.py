import warnings
warnings.filterwarnings("ignore")

import io
import hashlib
from math import ceil

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.pipeline import Pipeline

# local utils
from utils.guards import detect_problem_type, fingerprint, need_retrain, read_csv_with_limits
from utils.models import get_catalog
from utils.prep import build_preprocessor
from utils.viz import plot_confusion_matrix, plot_residuals, plot_parity
from utils.explain import shap_summary, lime_local

st.set_page_config(page_title="ExplainMyModel (MVP)", page_icon="🔍", layout="wide")
st.title("🔍 ExplainMyModel — MVP")
st.caption("Train valid models for your dataset and understand decisions with **SHAP** (global) and **LIME** (local).")

# ---------------------------
# Helpers
# ---------------------------
def suggest_target_column(df: pd.DataFrame) -> str:
    """Prefer a categorical column as the label; otherwise a low-cardinality numeric; else last column."""
    cat_like = [c for c in df.columns if str(df[c].dtype) in ("object","category","bool")]
    if cat_like:
        return cat_like[-1]
    num_like = [c for c in df.columns if c not in cat_like]
    best, best_card = None, 10**9
    for c in num_like:
        card = df[c].nunique(dropna=True)
        if 2 <= card <= 20 and card < best_card:
            best, best_card = c, card
    return best or df.columns[-1]

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("1) Data")
    use_demo = st.toggle("Use demo dataset", value=True)
    if use_demo:
        demo_choice = st.selectbox("Choose demo dataset", [
            "Classification — Iris",
            "Regression — California Housing (sample)"
        ])
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

    st.header("2) Settings")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", 0, 9999, 42, step=1)

# ---------------------------
# Load data
# ---------------------------
if use_demo:
    if "Classification" in demo_choice:
        df = pd.read_csv("assets/demo_classification.csv")   # Iris
    else:
        df = pd.read_csv("assets/demo_regression.csv")       # CA housing sample
else:
    if "uploaded" not in locals() or not uploaded:
        st.info("Upload a CSV or toggle a demo dataset to proceed.")
        st.stop()
    df = read_csv_with_limits(uploaded)

st.subheader("Dataset Preview")
st.write(df.head())
st.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# ---------------------------
# Target selection (smart default)
# ---------------------------
suggested_target = suggest_target_column(df)
target = st.selectbox(
    "Select target column",
    df.columns,
    index=list(df.columns).index(suggested_target)
)
y = df[target]
X = df.drop(columns=[target])

# try to coerce numeric-looking strings in X
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = pd.to_numeric(X[col], errors="ignore")

# ---------------------------
# Problem type (auto + override)
# ---------------------------
ptype = detect_problem_type(y)

with st.sidebar:
    task_override = st.selectbox(
        "Task type (override)",
        options=["Auto (detected)", "Classification", "Regression"],
        index=0
    )

if task_override == "Classification":
    ptype = "classification"
elif task_override == "Regression":
    ptype = "regression"

st.info(
    f"Detected problem type: **{ptype.capitalize()}**"
    + (" (forced override)" if task_override != "Auto (detected)" else "")
)

# nudge if looks like regression but we have obvious categoricals
if ptype == "regression":
    cat_small = [c for c in df.columns if c != target and str(df[c].dtype) in ("object","category","bool")]
    if cat_small:
        st.info(
            "This looks like a regression target. If you intended classification, "
            "try selecting a categorical label column (e.g., "
            + ", ".join(cat_small[:3]) + ") or use the **Task type (override)**."
        )

# ---------------------------
# Model selection
# ---------------------------
catalog = get_catalog(ptype)
model_name = st.selectbox("Model", list(catalog.keys()))
model = catalog[model_name]

# ---------------------------
# Preprocessor
# ---------------------------
pre = build_preprocessor(X)

# ---------------------------
# Safe stratified split (classification)
# ---------------------------
def safe_stratified_split(X, y, test_size, random_state):
    vc = y.value_counts(dropna=False)
    min_class = int(vc.min())
    if min_class < 2:
        st.warning(f"Stratification disabled: least populated class has only {min_class} sample.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)
    low = 1.0 / min_class
    high = 1.0 - (1.0 / min_class)
    if test_size < low or test_size > high:
        adj = float(np.clip(test_size, low + 1e-6, high - 1e-6))
        st.info(
            f"Adjusted test_size from {test_size:.3f} → {adj:.3f} "
            f"to satisfy stratification with min-class size = {min_class}."
        )
        test_size = adj
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# split
if ptype == "classification":
    X_train, X_test, y_train, y_test = safe_stratified_split(X, y, test_size, random_state)
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )

# ---------------------------
# Fingerprint & gating
# ---------------------------
fp = fingerprint(df, target, ptype, model_name)
if "trained_fp" not in st.session_state: st.session_state.trained_fp = None
if "pipe" not in st.session_state: st.session_state.pipe = None

need = (st.session_state.trained_fp != fp) or (st.session_state.pipe is None)
if need:
    st.warning("Dataset/target/model changed — retraining required.")
else:
    st.success("Model is in sync with this dataset.")

if st.button("Train / Retrain", type="primary"):
    pipe = Pipeline([("prep", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    st.session_state.pipe = pipe
    st.session_state.trained_fp = fp
    st.success("Training complete.")

if st.session_state.pipe is None or st.session_state.trained_fp != fp:
    st.info("Train the model to view metrics and explanations.")
    st.stop()

pipe = st.session_state.pipe
prep = pipe.named_steps["prep"]
mdl  = pipe.named_steps["model"]

# ---------------------------
# Performance
# ---------------------------
st.subheader("Performance")

if ptype == "classification":
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    c1, c2 = st.columns([1,1])
    with c1:
        st.metric("Accuracy", f"{acc:.3f}")
    with c2:
        labels = list(pd.Series(y).astype("category").cat.categories) if str(y.dtype)=="category" else sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        st.plotly_chart(plot_confusion_matrix(cm, labels), use_container_width=True)
    st.caption("Confusion Matrix: rows = actual, columns = predicted.")
    st.text(classification_report(y_test, y_pred))
else:
    y_pred = pipe.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    try:
        rmse = mean_squared_error(y_test, y_pred, squared=False)  # newer sklearn
    except TypeError:
        rmse = mean_squared_error(y_test, y_pred) ** 0.5          # older sklearn
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2:.3f}")
    c2.metric("RMSE", f"{rmse:.3f}")
    c3.metric("MAE", f"{mae:.3f}")
    st.plotly_chart(plot_residuals(y_test, y_pred), use_container_width=True)
    st.plotly_chart(plot_parity(y_test, y_pred), use_container_width=True)

# ---------------------------
# Global SHAP
# ---------------------------
st.subheader("Global Explanations (SHAP)")
feature_names_transformed = list(prep.get_feature_names_out())
st.caption("Beeswarm: each dot = a row; left/right shows how a feature pushes prediction down/up.")
shap_summary(pipe, X_train, X_test, ptype, feature_names_transformed)

# ---------------------------
# Local LIME
# ---------------------------
st.subheader("Local Explanation (LIME)")
row_idx = st.number_input("Pick a test row index", 0, int(len(X_test)-1), 0, step=1)
st.caption("Bars show feature contributions for this single prediction.")
feature_names_raw = list(X.columns)
lime_local(pipe, X_train, X_test, feature_names_raw, ptype, int(row_idx))

# ---------------------------
# Export
# ---------------------------
st.subheader("Export")
if hasattr(mdl, "feature_importances_"):
    importances = pd.Series(mdl.feature_importances_, index=feature_names_transformed).sort_values(ascending=False)
    export_df = importances.reset_index().rename(columns={"index": "feature", 0: "importance"})
    st.download_button(
        "Download Feature Importances (CSV)",
        export_df.to_csv(index=False).encode(),
        "feature_importances.csv",
        "text/csv"
    )
else:
    st.caption("No native feature importances for this model. Use the SHAP summary above for a global view.")

st.success("Done. Try a different model, target, or dataset!")
