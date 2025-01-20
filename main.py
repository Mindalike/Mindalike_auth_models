import gzip
import joblib
from fastapi import FastAPI, Response, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import logging
from datetime import datetime
from typing import Optional
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
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

# Load the model
try:
    with gzip.open('creditcard_fraud_model.joblib.gz', 'rb') as f:
        model = joblib.load(f)
        logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Initialize user patterns storage
PATTERNS_FILE = "user_patterns.json"
if not Path(PATTERNS_FILE).exists():
    with open(PATTERNS_FILE, "w") as f:
        json.dump({}, f)

def load_patterns():
    try:
        with open(PATTERNS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading patterns: {str(e)}")
        return {}

def save_patterns(patterns):
    try:
        with open(PATTERNS_FILE, "w") as f:
            json.dump(patterns, f)
    except Exception as e:
        logger.error(f"Error saving patterns: {str(e)}")

class LoginAttempt(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    timestamp: float = Field(..., description="Unix timestamp of the login attempt")
    ip_address: str = Field(..., description="IP address of the login attempt")
    device_type: str = Field(..., description="Type of device used (mobile, desktop, etc.)")
    browser: str = Field(..., description="Browser information")
    location: Optional[str] = Field(None, description="Geographic location if available")
    login_success: bool = Field(..., description="Whether the login was successful")

def extract_features(attempt: LoginAttempt, patterns: dict) -> list[float]:
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
        float(attempt.device_type not in user_patterns["usual_devices"]),  # New device?
        float(attempt.location not in user_patterns["usual_locations"]),  # New location?
        float(user_patterns["failed_attempts"]),  # Previous failed attempts
        float(user_patterns["total_attempts"]),  # Total login attempts
    ] + [0] * 25  # Padding to match model's expected 30 features
    
    return features

def update_patterns(attempt: LoginAttempt):
    patterns = load_patterns()
    user_patterns = patterns.get(attempt.user_id, {
        "usual_locations": [],
        "usual_devices": [],
        "usual_times": [],
        "failed_attempts": 0,
        "total_attempts": 0
    })
    
    # Update patterns
    if attempt.location and attempt.location not in user_patterns["usual_locations"]:
        user_patterns["usual_locations"].append(attempt.location)
    if attempt.device_type not in user_patterns["usual_devices"]:
        user_patterns["usual_devices"].append(attempt.device_type)
    
    hour = datetime.fromtimestamp(attempt.timestamp).hour
    if hour not in user_patterns["usual_times"]:
        user_patterns["usual_times"].append(hour)
    
    user_patterns["total_attempts"] += 1
    if not attempt.login_success:
        user_patterns["failed_attempts"] += 1
    
    patterns[attempt.user_id] = user_patterns
    save_patterns(patterns)

@app.post("/analyze_login", status_code=status.HTTP_200_OK)
async def analyze_login(attempt: LoginAttempt, response: Response):
    """
    Analyze a login attempt for suspicious patterns.
    Returns a risk score and recommended actions.
    """
    logger.info(f"Analyzing login attempt for user {attempt.user_id}")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Security model not available")
    
    # Load existing patterns
    patterns = load_patterns()
    
    # Extract features
    features = extract_features(attempt, patterns)
    
    try:
        # Get model prediction
        prediction = model.predict(np.array(features).reshape(1, -1))
        prediction_proba = getattr(model, 'predict_proba', lambda x: [[0.5, 0.5]])(np.array(features).reshape(1, -1))
        risk_score = float(prediction_proba[0][1])  # Probability of suspicious activity
        
        # Update patterns with this attempt
        update_patterns(attempt)
        
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
                "unusual_time": hour not in patterns.get(attempt.user_id, {}).get("usual_times", [])
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing login attempt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user_patterns/{user_id}", status_code=status.HTTP_200_OK)
async def get_user_patterns(user_id: str, response: Response):
    """
    Get the stored patterns for a specific user.
    """
    patterns = load_patterns()
    if user_id not in patterns:
        raise HTTPException(status_code=404, detail="User not found")
    return patterns[user_id]
