# utils/predict.py

import joblib
import os
import numpy as np
import pandas as pd

# Load trained model and preprocessing tools
model_path = os.path.join("ml_model", "model.pkl")
model, selector, symptom_list, label_encoder = joblib.load(model_path)

# Load supporting CSVs
description_df = pd.read_csv("data/description.csv")
medications_df = pd.read_csv("data/medications.csv")
precautions_df = pd.read_csv("data/precautions_df.csv").drop(columns=['Unnamed: 0'], errors='ignore')

def encode_symptoms(symptoms):
    """
    Converts selected symptoms into a binary vector.
    """
    vector = np.zeros(len(symptom_list))
    for symptom in symptoms:
        if symptom in symptom_list:
            idx = symptom_list.index(symptom)
            vector[idx] = 1
    return vector

def predict_disease(symptoms):
    """
    Predicts disease based on symptoms and returns disease details.
    """
    # Encode symptoms and apply feature selection
    encoded_vector = encode_symptoms(symptoms)
    selected_features = selector.transform([encoded_vector])

    # Make prediction
    pred_index = model.predict(selected_features)[0]
    disease = label_encoder.inverse_transform([pred_index])[0]

    # Retrieve details
    desc = description_df[description_df['Disease'] == disease]['Description'].values[0]
    meds = medications_df[medications_df['Disease'] == disease]['Medication'].values[0]
    precautions = precautions_df[precautions_df['Disease'] == disease].iloc[:, 1:].values.flatten()
    precautions = [p for p in precautions if p != "None" and pd.notna(p)]

    return {
        "disease": disease,
        "description": desc,
        "medications": meds,
        "precautions": precautions
    }
