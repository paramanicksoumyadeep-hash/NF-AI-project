import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("data/raw_data.csv")

le = LabelEncoder()
df["material"] = le.fit_transform(df["material"])

# Save encoder
joblib.dump(le, "models/material_encoder.pkl")

df.to_csv("data/processed_data.csv", index=False)

print("Preprocessing complete!")
