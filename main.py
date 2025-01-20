import gzip
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI with additional configuration
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting fraudulent credit card transactions",
    version="1.0.0"
)

# Load the compressed model
try:
    with gzip.open('creditcard_fraud_model.joblib.gz', 'rb') as f:
        model = joblib.load(f)
        logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Define the input schema
class FraudPredictionRequest(BaseModel):
    features: list[float]

# Define root endpoint
@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed")
    return {
        "status": "online",
        "message": "Welcome to the Credit Card Fraud Detection API",
        "endpoints": {
            "/": "This help message",
            "/predict": "Make fraud predictions (POST request)"
        }
    }

# Define the prediction endpoint
@app.post("/predict")
async def predict(request: FraudPredictionRequest):
    logger.info("Prediction endpoint accessed")
    if model is None:
        logger.error("Model not loaded")
        return {"error": "Model not loaded"}
    
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)
        logger.info(f"Prediction made: {prediction[0]}")
        return {"prediction": int(prediction[0])}
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return {"error": str(e)}
