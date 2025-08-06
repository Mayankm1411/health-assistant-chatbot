# utils/predict.py

import joblib
import base64
import os
import io
import numpy as np
import pandas as pd

# Path to base64 model text
base64_path = os.path.join("ml_model", "model_base64.txt")

# Decode and load model
with open(base64_path, "r") as f:
    base64_data = f.read()

model_bytes = base64.b64decode(base64_data)
model_file = io.BytesIO(model_bytes)
model, selector, symptom_list, label_encoder = joblib.load(model_file)

# Load metadata
description_df = pd.read_csv("data/description.csv")
medications_df = pd.read_csv("data/medications.csv")
precautions_df = pd.read_csv("data/precautions_df.csv").drop(columns=["Unnamed: 0"], errors="ignore")

# One-hot encoding
def encode_symptoms(symptoms):
    vector = np.zeros(len(symptom_list))
    for s in symptoms:
        if s in symptom_list:
            idx = symptom_list.index(s)
            vector[idx] = 1
    return vector

# Main prediction function
def predict_disease(symptoms):
    encoded = encode_symptoms(symptoms)
    selected = selector.transform([encoded])
    pred_index = model.predict(selected)[0]
    disease = label_encoder.inverse_transform([pred_index])[0]

    # Details
    desc = description_df[description_df['Disease'] == disease]['Description'].values[0]
    meds = medications_df[medications_df['Disease'] == disease]['Medication'].values[0]
    precautions = precautions_df[precautions_df['Disease'] == disease].iloc[:, 1:].values.flatten()
    precautions = [p for p in precautions if p and p != "None"]

    return {
        "disease": disease,
        "description": desc,
        "medications": meds,
        "precautions": precautions
    }
