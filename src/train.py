from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

def train_model(X_train, y_train):
    np.random.seed(42)
    pipelines = {
        "random_forest": Pipeline([("scalar", StandardScaler()), ("model",RandomForestClassifier(n_estimators=100))]),
        "logistic_regression": Pipeline([("scalar", StandardScaler()), ("model", LogisticRegression())])
    }
    trained_models = {}

    for name, model in pipelines.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models


def save_model(model, path="models/model.pkl"):
    joblib.dump(model, path)
