from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
import joblib

def train_model(X_train, y_train):
    pipelines = {
        "random_forest": Pipeline([("scalar", StandardScaler()), ("model",RandomForestClassifier(n_estimators=100, random_state=42))]),
        "logistic_regression": Pipeline([("scalar", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))])
    }
    trained_models = {}

    for name, model in pipelines.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models


def save_model(model, path="models/model.pkl"):
    joblib.dump(model, path)
