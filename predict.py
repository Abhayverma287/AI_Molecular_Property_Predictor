import joblib
import numpy as np

from src.feature_engineering import smiles_to_fingerprint

model = joblib.load("models/property_model.pkl")

def predict_property(smiles):

    fp = smiles_to_fingerprint(smiles)

    fp = np.array(fp).reshape(1, -1)

    pred = model.predict(fp)[0]

    if pred == 1:
        return "⚠️ Potentially Toxic"
    else:
        return "✅ Low Toxicity"