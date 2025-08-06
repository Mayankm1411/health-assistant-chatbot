import pandas as pd

# Load the full symptoms dataset
df = pd.read_csv("data/symptoms_df.csv")

# Extract all symptom columns
symptom_columns = [col for col in df.columns if col.startswith("Symptom")]

# Flatten and clean up
symptoms = pd.unique(df[symptom_columns].values.ravel())
symptoms = [s for s in symptoms if isinstance(s, str) and s.lower() != "none"]

# Save to a new CSV
pd.DataFrame(symptoms, columns=["Symptom"]).to_csv("data/symptoms.csv", index=False)

print(f"✅ Created symptoms.csv with {len(symptoms)} unique symptoms.")
