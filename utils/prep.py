from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def split_cols(X):
    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("object","category","bool")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols

def build_preprocessor(X):
    cat_cols, num_cols = split_cols(X)
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False))
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return pre
