import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

def _to_matrix(vals):
    """
    Normalize SHAP outputs to a single (n_samples, n_features) matrix:
    - Explanation -> .values
    - list of Explanation -> stack
    - list of arrays -> stack
    - array -> as-is
    Returns (matrix, is_list, n_classes)
    """
    try:
        # Newer SHAP: Explanation
        from shap import Explanation  # type: ignore
        ExplanationType = Explanation
    except Exception:
        ExplanationType = tuple()  # nothing

    if isinstance(vals, ExplanationType):
        return vals.values, False, 1

    if isinstance(vals, list) and len(vals) > 0:
        # list of Explanation or list of arrays
        first = vals[0]
        if isinstance(first, ExplanationType):
            mats = [v.values for v in vals]
        else:
            mats = [np.array(v) for v in vals]
        # Expect shape (n_classes, n_samples, n_features)
        return np.array(mats), True, len(mats)

    # plain array
    arr = np.array(vals)
    return arr, False, 1

def _plot_summary(vals_matrix, data_matrix, feature_names):
    fig, _ = plt.subplots()
    shap.summary_plot(vals_matrix, data_matrix, feature_names=feature_names, show=False)
    st.pyplot(fig, clear_figure=True)

def shap_summary(pipe, X_train, X_test, problem_type: str, feature_names_transformed):
    """
    Global SHAP summary using the FULL sklearn Pipeline (prep + model).
    - Uses shap.Explainer with the pipeline's predict/predict_proba callable.
    - Works for binary & multiclass:
        * binary  -> plots class 1
        * multiclass -> plots mean(|SHAP|) across classes
    - Samples data for speed.
    NOTE: feature_names_transformed is unused here (we plot in RAW feature space).
    """
    # --- choose prediction function from the full pipeline ---
    if problem_type == "classification" and hasattr(pipe, "predict_proba"):
        def predict_fn(X):
            return pipe.predict_proba(X)
    else:
        def predict_fn(X):
            return pipe.predict(X)

    # --- sample background and test for performance ---
    n_bg = min(300, len(X_train))
    n_sm = min(300, len(X_test))
    background = X_train.iloc[:n_bg].copy()
    sample     = X_test.iloc[:n_sm].copy()

    try:
        # Modern, model-agnostic API
        explainer = shap.Explainer(predict_fn, background)
        sv = explainer(sample)  # sv can be Explanation or list/array depending on SHAP version/model

        # --- normalize to a single (n_samples, n_features) matrix ---
        # Newer SHAP returns an Explanation with .values possibly (n, f) or (n, f, n_classes)
        values = getattr(sv, "values", sv)

        # Case A: multi-output with class axis at the end → (n, f, C)
        if isinstance(values, np.ndarray) and values.ndim == 3:
            n_classes = values.shape[2]
            if problem_type == "classification":
                if n_classes == 2:
                    matrix = values[:, :, 1]          # positive class
                else:
                    matrix = np.mean(np.abs(values), axis=2)  # multiclass aggregate
            else:
                matrix = np.mean(values, axis=2)     # defensive
            features_for_plot = sample

        # Case B: SHAP returned a list (older TreeExplainer path)
        elif isinstance(values, list) and len(values) > 0:
            mats = [np.array(v) for v in values]     # list of (n, f)
            if problem_type == "classification":
                if len(mats) == 2:
                    matrix = mats[1]
                else:
                    matrix = np.mean(np.abs(np.stack(mats, axis=0)), axis=0)
            else:
                matrix = np.mean(np.stack(mats, axis=0), axis=0)
            features_for_plot = sample

        # Case C: already a single matrix (n, f)
        else:
            matrix = np.array(values)
            features_for_plot = sample

        st.caption(f"SHAP computed on {matrix.shape[0]} rows × {matrix.shape[1]} features.")
        fig, _ = plt.subplots()
        shap.summary_plot(matrix, features_for_plot, feature_names=list(sample.columns), show=False)
        st.pyplot(fig, clear_figure=True)

    except Exception as e:
        st.warning(f"SHAP summary skipped: {e}")

def lime_local(pipe,
               X_train_raw: pd.DataFrame,
               X_test_raw: pd.DataFrame,
               feature_names_raw: list[str],
               problem_type: str,
               row_idx: int):
    """
    LIME in RAW feature space with strict numeric matrix and categorical handling:
    - LIME sees numeric arrays only (categoricals encoded as integer codes)
    - numeric columns coerced to numeric; NaNs filled with train medians
    - predict_fn decodes codes -> strings and routes through sklearn Pipeline
    """
    from lime.lime_tabular import LimeTabularExplainer

    # Clean indices
    X_train_raw = X_train_raw.reset_index(drop=True)
    X_test_raw  = X_test_raw.reset_index(drop=True)

    mode = "classification" if problem_type == "classification" else "regression"

    # Detect categorical vs numeric columns
    cat_cols = [c for c in feature_names_raw
                if str(X_train_raw[c].dtype) in ("object", "category", "bool")]
    num_cols = [c for c in feature_names_raw if c not in cat_cols]
    cat_idx  = [feature_names_raw.index(c) for c in cat_cols]

    # Build code maps for categoricals using TRAIN values
    cat_to_code: dict[str, dict[str, int]] = {}
    code_to_cat: dict[str, dict[int, str]] = {}
    categorical_names: dict[int, list[str]] = {}

    for col in cat_cols:
        vals = pd.Series(X_train_raw[col]).astype(str).fillna("NA").unique().tolist()
        vals = sorted(vals)  # deterministic
        mapping = {v: i for i, v in enumerate(vals)}
        cat_to_code[col] = mapping
        code_to_cat[col] = {i: v for v, i in mapping.items()}
        categorical_names[feature_names_raw.index(col)] = vals

    # Precompute numeric medians (LIME-only fillna)
    num_medians: dict[str, float] = {}
    for col in num_cols:
        col_numeric = pd.to_numeric(X_train_raw[col], errors="coerce")
        num_medians[col] = float(col_numeric.median()) if not col_numeric.dropna().empty else 0.0

    def _encode_df(df: pd.DataFrame) -> pd.DataFrame:
        """Encode categoricals -> int codes; numerics -> coerced & filled; preserve column order."""
        df2 = df.copy()

        # numerics: coerce & fill NaN with train medians
        for c in num_cols:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
            df2[c] = df2[c].fillna(num_medians.get(c, 0.0))

        # categoricals: codes; unseen -> UNK code
        for col in cat_cols:
            s = df2[col].astype(str).fillna("NA")
            mapping = cat_to_code[col]
            unk_code = len(mapping)
            df2[col] = s.map(mapping).fillna(unk_code).astype(int)

        return df2[feature_names_raw]

    def _decode_arr(arr: np.ndarray) -> pd.DataFrame:
        """Decode integer-coded categoricals back to strings before passing to Pipeline."""
        df = pd.DataFrame(arr, columns=feature_names_raw)

        # keep numerics numeric
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # decode categoricals, unknowns -> "UNK"
        for col in cat_cols:
            idx = feature_names_raw.index(col)
            inv = code_to_cat[col]
            df[col] = pd.Series(df[col]).round(0).astype(int).map(inv).fillna("UNK")

        return df

    # LIME training matrix (strictly numeric float, no NaNs)
    X_train_enc = _encode_df(X_train_raw).to_numpy(dtype=float)

    # Predictor: decode LIME's perturbed arrays -> Pipeline
    if mode == "classification" and hasattr(pipe, "predict_proba"):
        def predict_fn(arr):
            df_decoded = _decode_arr(arr)
            return pipe.predict_proba(df_decoded)
    else:
        def predict_fn(arr):
            df_decoded = _decode_arr(arr)
            return pipe.predict(df_decoded)

    # Initialize LIME
    explainer = LimeTabularExplainer(
        training_data=X_train_enc,
        feature_names=feature_names_raw,
        categorical_features=cat_idx if cat_idx else None,
        categorical_names=categorical_names if cat_idx else None,
        discretize_continuous=True,
        mode=mode
    )

    # Instance to explain (encoded, numeric, float)
    inst_raw = X_test_raw.iloc[int(row_idx)][feature_names_raw].to_frame().T
    inst_enc = _encode_df(inst_raw).to_numpy(dtype=float).ravel()

    exp = explainer.explain_instance(inst_enc, predict_fn, num_features=10)
    fig = exp.as_pyplot_figure()
    st.pyplot(fig, clear_figure=True)
