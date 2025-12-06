# utils/guards.py
import io
import hashlib
import numpy as np
import pandas as pd
import streamlit as st

MAX_MB = 10
MAX_ROWS = 50000
MAX_COLS = 100

def detect_problem_type(y: pd.Series, max_class_card: int = 20) -> str:
    """
    Heuristic:
    - If dtype is object/category/bool -> classification
    - If numeric:
        * If unique values <= max_class_card AND values are mostly integer-like -> classification
        * Else -> regression
    Fixes cases like Iris and integer-coded labels.
    """
    dtype = str(y.dtype)
    if dtype in ("category", "object", "bool"):
        return "classification"

    uniq = y.nunique(dropna=True)
    if uniq <= max_class_card:
        sample = pd.to_numeric(y.dropna(), errors="coerce")
        if not sample.empty:
            frac_int_like = (np.isclose(sample, np.round(sample))).mean()
            if frac_int_like >= 0.95:
                return "classification"
    return "regression"

def fingerprint(df: pd.DataFrame, target: str, ptype: str, model_name: str) -> str:
    sig = (
        str(list(df.columns)) +
        str(df.dtypes.astype(str).tolist()) +
        target + ptype + model_name
    )
    return hashlib.md5(sig.encode()).hexdigest()

def need_retrain(ss, fp: str) -> bool:
    return (ss.get("trained_fp") != fp) or (ss.get("pipe") is None)

def read_csv_with_limits(uploaded) -> pd.DataFrame:
    # Size gate
    uploaded.seek(0, io.SEEK_END)
    mb = uploaded.tell() / (1024 * 1024)
    uploaded.seek(0)
    if mb > MAX_MB:
        st.error(f"File too large ({mb:.1f} MB). Max allowed is {MAX_MB} MB.")
        st.stop()

    df = pd.read_csv(uploaded)

    # Row/col caps
    if df.shape[0] > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42)
        st.warning(f"Dataset sampled to {MAX_ROWS} rows for performance.")
    if df.shape[1] > MAX_COLS:
        st.error(f"Too many columns ({df.shape[1]}). Max allowed is {MAX_COLS}.")
        st.stop()

    # High-cardinality categorical warning
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) == "category"]
    hi_card = [c for c in cat_cols if df[c].nunique() > 100]
    if hi_card:
        st.info(
            f"High-cardinality categorical columns detected: {hi_card[:5]} "
            f"{'(and more...)' if len(hi_card) > 5 else ''}. One-hot may get large."
        )

    # Simple PII hint
    pii_like = [c for c in df.columns if c.lower() in {"name", "email", "phone", "mobile", "aadhaar", "ssn"}]
    if pii_like:
        st.warning(f"Potential PII columns detected: {pii_like}. Data stays local; remove if not needed.")

    return df
