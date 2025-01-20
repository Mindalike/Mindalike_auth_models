import gzip
import joblib
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# Define root endpoint with HEAD support
@app.get("/", status_code=status.HTTP_200_OK)
@app.head("/", status_code=status.HTTP_200_OK)
async def read_root(response: Response):
    logger.info("Root endpoint accessed")
    return {
        "status": "online",
        "message": "Welcome to the Credit Card Fraud Detection API",
        "model_loaded": model is not None,
        "endpoints": {
            "/": "This help message",
            "/docs": "Interactive API documentation",
            "/predict": "Make fraud predictions (POST request)"
        }
    }

# Define the prediction endpoint
@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(request: FraudPredictionRequest, response: Response):
    logger.info("Prediction endpoint accessed")
    if model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"error": "Model not loaded"}
    
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)
        logger.info(f"Prediction made: {prediction[0]}")
        return {"prediction": int(prediction[0])}
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": str(e)}
