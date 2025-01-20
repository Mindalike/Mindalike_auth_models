import gzip
import joblib
from fastapi import FastAPI, Response, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
import numpy as np
import logging
from datetime import datetime
import json
from pathlib import Path

# Configure more detailed logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI with additional configuration
app = FastAPI(
    title="Login Security API",
    description="API for detecting suspicious login patterns",
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

# Load the model with more robust error handling
model = None
try:
    with gzip.open('creditcard_fraud_model.joblib.gz', 'rb') as f:
        model = joblib.load(f)
        logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Critical error loading model: {str(e)}")

# Initialize user patterns storage
PATTERNS_FILE = "user_patterns.json"
Path(PATTERNS_FILE).touch(exist_ok=True)

def safe_load_json(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def safe_save_json(data, file_path):
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving JSON: {str(e)}")

class LoginAttempt(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=50, description="Unique identifier for the user")
    timestamp: float = Field(..., description="Unix timestamp of the login attempt")
    ip_address: str = Field(..., description="IP address of the login attempt")
    device_type: str = Field(..., description="Type of device used (mobile, desktop, etc.)")
    browser: str = Field(..., description="Browser information")
    location: Optional[str] = Field(None, description="Geographic location if available")
    login_success: bool = Field(..., description="Whether the login was successful")

def extract_features(attempt: LoginAttempt, patterns: dict) -> List[float]:
    try:
        user_patterns = patterns.get(attempt.user_id, {
            "usual_locations": [],
            "usual_devices": [],
            "usual_times": [],
            "failed_attempts": 0,
            "total_attempts": 0
        })
        
        # Convert timestamp to hour of day (0-23)
        hour = datetime.fromtimestamp(attempt.timestamp).hour
        
        # Calculate features
        features = [
            float(hour),  # Time of day
            float(attempt.device_type not in user_patterns.get("usual_devices", [])),  # New device?
            float(attempt.location not in user_patterns.get("usual_locations", [])),  # New location?
            float(user_patterns.get("failed_attempts", 0)),  # Previous failed attempts
            float(user_patterns.get("total_attempts", 0)),  # Total login attempts
        ] + [0.0] * 25  # Padding to match model's expected 30 features
        
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise

def update_user_patterns(attempt: LoginAttempt):
    try:
        patterns = safe_load_json(PATTERNS_FILE)
        user_patterns = patterns.get(attempt.user_id, {
            "usual_locations": [],
            "usual_devices": [],
            "usual_times": [],
            "failed_attempts": 0,
            "total_attempts": 0
        })
        
        # Update patterns
        if attempt.location and attempt.location not in user_patterns.get("usual_locations", []):
            user_patterns.setdefault("usual_locations", []).append(attempt.location)
        if attempt.device_type not in user_patterns.get("usual_devices", []):
            user_patterns.setdefault("usual_devices", []).append(attempt.device_type)
        
        hour = datetime.fromtimestamp(attempt.timestamp).hour
        if hour not in user_patterns.get("usual_times", []):
            user_patterns.setdefault("usual_times", []).append(hour)
        
        user_patterns["total_attempts"] = user_patterns.get("total_attempts", 0) + 1
        if not attempt.login_success:
            user_patterns["failed_attempts"] = user_patterns.get("failed_attempts", 0) + 1
        
        patterns[attempt.user_id] = user_patterns
        safe_save_json(patterns, PATTERNS_FILE)
    except Exception as e:
        logger.error(f"Error updating user patterns: {str(e)}")
        raise

@app.post("/analyze_login", status_code=status.HTTP_200_OK)
async def analyze_login(attempt: LoginAttempt, response: Response):
    """
    Analyze a login attempt for suspicious patterns.
    Returns a risk score and recommended actions.
    """
    try:
        logger.info(f"Analyzing login attempt for user {attempt.user_id}")
        
        if model is None:
            raise HTTPException(status_code=503, detail="Security model not available")
        
        # Load existing patterns
        patterns = safe_load_json(PATTERNS_FILE)
        
        # Extract features
        features = extract_features(attempt, patterns)
        
        # Get model prediction
        prediction = model.predict(np.array(features).reshape(1, -1))
        prediction_proba = getattr(model, 'predict_proba', lambda x: [[0.5, 0.5]])(np.array(features).reshape(1, -1))
        risk_score = float(prediction_proba[0][1])  # Probability of suspicious activity
        
        # Update patterns with this attempt
        update_user_patterns(attempt)
        
        # Determine required actions based on risk score
        required_actions = []
        if risk_score > 0.8:
            required_actions.append("require_2fa")
            required_actions.append("notify_admin")
        elif risk_score > 0.5:
            required_actions.append("require_2fa")
        elif risk_score > 0.3:
            required_actions.append("monitor_closely")
        
        return {
            "risk_score": risk_score,
            "is_suspicious": bool(prediction[0]),
            "required_actions": required_actions,
            "unusual_patterns": {
                "new_device": attempt.device_type not in patterns.get(attempt.user_id, {}).get("usual_devices", []),
                "new_location": attempt.location not in patterns.get(attempt.user_id, {}).get("usual_locations", []),
                "unusual_time": datetime.fromtimestamp(attempt.timestamp).hour not in patterns.get(attempt.user_id, {}).get("usual_times", [])
            }
        }
    except ValidationError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error in login analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/user_patterns/{user_id}", status_code=status.HTTP_200_OK)
async def get_user_patterns(user_id: str, response: Response):
    """
    Get the stored patterns for a specific user.
    """
    patterns = safe_load_json(PATTERNS_FILE)
    if user_id not in patterns:
        raise HTTPException(status_code=404, detail="User not found")
    return patterns[user_id]
