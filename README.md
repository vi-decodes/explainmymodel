# ExplainMyModel (MVP)

An interactive Streamlit app to **train valid models for your dataset** (classification or regression),
and **explain** them with **SHAP** (global) and **LIME** (local), with **guardrails** that prevent invalid
model–data pairings and stale explanations.

## Features (MVP)
- Upload CSV or use built-in demo datasets
- Auto-detect **problem type** (classification vs regression)
- Show **only valid models** for the detected task
- Automatic preprocessing: impute missing values, scale numerics, one-hot encode categoricals
- Metrics & visuals:
  - Classification: Accuracy, Confusion Matrix, Classification Report
  - Regression: R², RMSE, MAE, Residuals, Predicted vs Actual (parity)
- **Global explanations** (SHAP summary) + **Local explanations** (LIME)
- **State fingerprint** locks explanations to the current dataset/target/model (no stale results)
- Export CSV of importances (or SHAP mean |value|)

## Quick start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
