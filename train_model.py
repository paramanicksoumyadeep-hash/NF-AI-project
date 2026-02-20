import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("data/processed_data.csv")

X = df.drop(["flux", "rejection"], axis=1)

y_flux = df["flux"]
y_rej = df["rejection"]

X_train, X_test, y_flux_train, y_flux_test = train_test_split(
    X, y_flux, test_size=0.2, random_state=42
)

_, _, y_rej_train, y_rej_test = train_test_split(
    X, y_rej, test_size=0.2, random_state=42
)

flux_model = RandomForestRegressor(n_estimators=200)
flux_model.fit(X_train, y_flux_train)

rej_model = RandomForestRegressor(n_estimators=200)
rej_model.fit(X_train, y_rej_train)

joblib.dump(flux_model, "models/flux_model.pkl")
joblib.dump(rej_model, "models/rejection_model.pkl")

print("Models trained and saved!")
