from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def get_catalog(problem_type: str):
    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Decision Tree (Classifier)": DecisionTreeClassifier(random_state=42),
            "Random Forest (Classifier)": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        }
    else:
        return {
            "Linear Regression": LinearRegression(),
            "Decision Tree (Regressor)": DecisionTreeRegressor(random_state=42),
            "Random Forest (Regressor)": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        }
