import numpy as np
import joblib

def load_model(path="models/model.joblib"):
    return joblib.load(path)

def predict(input_data):
    model = load_model()
    prediction = model.predict(input_data)
    return prediction