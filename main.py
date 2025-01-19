from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved model
model = joblib.load('creditcard_fraud_model.joblib')

# Define the input schema
class FraudPredictionRequest(BaseModel):
    features: list[float]

# Initialize FastAPI
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
def predict(request: FraudPredictionRequest):
    features = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}