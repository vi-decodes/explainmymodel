import plotly.express as px
import pandas as pd

def plot_confusion_matrix(cm, labels):
    fig = px.imshow(
        cm, text_auto=True, x=labels, y=labels,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title="Confusion Matrix"
    )
    return fig

def plot_residuals(y_true, y_pred):
    resid = y_true - y_pred
    fig = px.histogram(resid, nbins=30, title="Residuals Histogram")
    return fig

def plot_parity(y_true, y_pred):
    df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    fig = px.scatter(df, x="Actual", y="Predicted", title="Predicted vs Actual (Parity Plot)")
    return fig
