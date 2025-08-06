import joblib
import os
import numpy as np
import pandas as pd

# Load model + files
model_path = os.path.join("ml_model", "model.pkl")
model, selector, symptom_list, label_encoder = joblib.load(model_path)

# Read additional info
description_df = pd.read_csv("data/description.csv")
medications_df = pd.read_csv("data/medications.csv")
precautions_df = pd.read_csv("data/precautions_df.csv").drop(columns=['Unnamed: 0'], errors='ignore')

def encode_symptoms(symptoms):
    vector = np.zeros(len(symptom_list))
    for s in symptoms:
        if s in symptom_list:
            idx = symptom_list.index(s)
            vector[idx] = 1
    return vector

def predict_disease(symptoms):
    encoded = encode_symptoms(symptoms)
    selected = selector.transform([encoded])
    pred_index = model.predict(selected)[0]
    disease = label_encoder.inverse_transform([pred_index])[0]

    # Get details
    desc = description_df[description_df['Disease'] == disease]['Description'].values[0]
    meds = medications_df[medications_df['Disease'] == disease]['Medication'].values[0]
    precautions = precautions_df[precautions_df['Disease'] == disease].iloc[:, 1:].values.flatten()
    precautions = [p for p in precautions if p != "None"]

    return {
        "disease": disease,
        "description": desc,
        "medications": meds,
        "precautions": precautions
    }
