import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(path):
    df= pd.read_csv(path)
    return df

def preprocess_data(df):
    X=df.drop("target", axis=1)
    y=df["target"]
    return X,y

def split_data(X,y):
    np.random.seed(42)
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)
    return X_train, X_test, y_train, y_test

