import gzip
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Load the compressed model
with gzip.open('creditcard_fraud_model.joblib.gz', 'rb') as f:
    model = joblib.load(f)

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
