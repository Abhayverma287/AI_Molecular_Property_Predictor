import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import smiles_to_fingerprint


df = pd.read_csv("data/molecules.csv")

X = []
y = []

for _, row in df.iterrows():

    fp = smiles_to_fingerprint(row["smiles"])

    X.append(fp)
    y.append(row["property"])


X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = RandomForestClassifier(
    n_estimators=200
)

model.fit(X_train, y_train)

joblib.dump(
    model,
    "models/property_model.pkl"
)

print("Model trained successfully")