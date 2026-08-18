from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Heart Disease Prediction API")

# Trained model load karna
model = joblib.load("heart_disease_model.pkl")

class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def home():
    return {"status": "Active", "message": "Heart Disease Prediction API running!"}

@app.post("/predict")
def predict_heart_disease(patient: PatientData):
    # Pydantic data ko dict se DataFrame mein convert karna
    input_df = pd.DataFrame([patient.model_dump()])
    
    # Model predictions
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    return {
        "prediction": int(prediction),
        "result": "Heart Disease Detected" if prediction == 1 else "Normal / No Heart Disease",
        "confidence_percentage": round(float(probability[1 if prediction == 1 else 0]) * 100, 2)
    }