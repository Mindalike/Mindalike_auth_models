import gzip
import joblib
from fastapi import FastAPI, Response, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Any
import numpy as np
import sys
import os
import traceback
from datetime import datetime, timedelta
import json
from pathlib import Path
import re

# Simplified logging with print statements
def debug_log(message):
    print(f"DEBUG: {message}")
    # Optionally, you can also write to a file
    try:
        with open('debug_login.log', 'a') as f:
            f.write(f"{message}\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")

def safe_log(message):
    """
    Safe logging function that prints and attempts to write to a log file
    """
    print(message)
    try:
        with open('login_security_debug.txt', 'a') as f:
            f.write(f"{message}\n")
    except Exception as log_error:
        print(f"Log writing error: {log_error}")

import os
import json
from datetime import datetime

def debug_to_file(message, filename='debug_login.jsonl'):
    """
    Write debug information to a JSON Lines file with timestamp
    """
    try:
        # Use an absolute path in the user's home directory
        debug_dir = os.path.expanduser('~/Desktop/debug_logs')
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, filename)
        
        debug_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": str(message)  # Ensure message is a string
        }
        
        # Append to file, creating if it doesn't exist
        with open(debug_path, 'a') as f:
            f.write(json.dumps(debug_entry) + '\n')
        
        # Optional: print to console as a backup
        print(f"DEBUG: {message}")
    
    except Exception as e:
        print(f"Error writing debug info: {e}")
        print(f"Original message: {message}")

# Initialize FastAPI with additional configuration
app = FastAPI(
    title="Advanced Login Security API",
    description="API for detecting sophisticated login patterns and potential security threats",
    version="1.1.0"
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
print("Starting model initialization...")
model = None
try:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    with gzip.open('creditcard_fraud_model.joblib.gz', 'rb') as f:
        import joblib
        model = joblib.load(f)
        print(f" Model loaded successfully: {type(model)}")
        
        # Attempt to verify model compatibility
        try:
            # Try a simple prediction to ensure model works
            test_features = np.zeros((1, 30))  # Assuming 30 features
            model.predict(test_features)
            print(" Model prediction test passed.")
        except Exception as pred_error:
            print(f" Model prediction test failed: {pred_error}")
            raise
except Exception as e:
    print(f" CRITICAL MODEL LOAD ERROR: {e}")
    print(traceback.format_exc())

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
        debug_log(f"Error saving JSON: {str(e)}")

class LoginAttempt(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=50, description="Unique identifier for the user")
    timestamp: float = Field(..., description="Unix timestamp of the login attempt")
    ip_address: str = Field(..., description="IP address of the login attempt")
    device_type: str = Field(..., description="Type of device used (mobile, desktop, etc.)")
    browser: str = Field(..., description="Browser information")
    location: Optional[str] = Field(None, description="Geographic location if available")
    login_success: bool = Field(..., description="Whether the login was successful")

def calculate_risk_factors(attempt, patterns):
    """
    Enhanced risk factor calculation with detailed file logging
    """
    try:
        debug_to_file(f"Calculating risk for user: {attempt.user_id}")
        debug_to_file(f"Attempt details: {attempt}")
        
        user_patterns = patterns.get(attempt.user_id, {
            "usual_locations": [],
            "usual_devices": [],
            "usual_times": [],
            "usual_browsers": [],
            "failed_attempts": 0,
            "total_attempts": 0,
            "last_successful_login": None
        })
        
        debug_to_file(f"Existing user patterns: {user_patterns}")
        
        risk_factors = {
            "location_risk": 0.0,
            "device_risk": 0.0,
            "browser_risk": 0.0,
            "time_risk": 0.0,
            "failed_attempt_risk": 0.0
        }
        
        # Location Risk
        if attempt.location not in user_patterns.get("usual_locations", []):
            risk_factors["location_risk"] = 0.3
            debug_to_file(f"New location detected: {attempt.location}")
        
        # Device Risk
        if attempt.device_type not in user_patterns.get("usual_devices", []):
            risk_factors["device_risk"] = 0.2
            debug_to_file(f"New device type detected: {attempt.device_type}")
        
        # Browser Risk
        if attempt.browser not in user_patterns.get("usual_browsers", []):
            risk_factors["browser_risk"] = 0.2
            debug_to_file(f"New browser detected: {attempt.browser}")
        
        # Time Risk (unusual hours or large time gaps)
        current_time = datetime.fromtimestamp(attempt.timestamp)
        last_login_time = user_patterns.get("last_successful_login")
        
        if last_login_time:
            time_diff = current_time - datetime.fromtimestamp(last_login_time)
            hour_diff = abs(current_time.hour - datetime.fromtimestamp(last_login_time).hour)
            
            debug_to_file(f"Time since last login: {time_diff}")
            debug_to_file(f"Hour difference: {hour_diff}")
            
            if time_diff > timedelta(days=7):
                risk_factors["time_risk"] = 0.3
                debug_to_file("Large time gap detected (>7 days)")
            elif hour_diff > 6:
                risk_factors["time_risk"] = 0.2
                debug_to_file("Unusual login hour detected")
        
        # Failed Attempt Risk
        failed_attempts = user_patterns.get("failed_attempts", 0)
        debug_to_file(f"Number of previous failed attempts: {failed_attempts}")
        
        if failed_attempts > 3:
            risk_factors["failed_attempt_risk"] = 0.3
            debug_to_file("Multiple failed attempts detected")
        elif failed_attempts > 1:
            risk_factors["failed_attempt_risk"] = 0.2
            debug_to_file("Some previous failed attempts")
        
        total_risk = sum(risk_factors.values())
        debug_to_file(f"Calculated risk factors: {risk_factors}")
        debug_to_file(f"Total risk score: {total_risk}")
        
        return risk_factors
    
    except Exception as e:
        debug_to_file(f"Risk calculation error: {e}")
        debug_to_file(traceback.format_exc())
        return {
            "location_risk": 0.0,
            "device_risk": 0.0,
            "browser_risk": 0.0,
            "time_risk": 0.0,
            "failed_attempt_risk": 0.0
        }

def extract_features(attempt, patterns):
    """
    Enhanced feature extraction with detailed file logging
    """
    try:
        debug_to_file(f"Extracting features for user: {attempt.user_id}")
        
        risk_factors = calculate_risk_factors(attempt, patterns)
        
        # Timestamp features
        hour = datetime.fromtimestamp(attempt.timestamp).hour
        total_risk = sum(risk_factors.values())
        
        features = [
            float(hour),  # Time of day
            float(risk_factors["location_risk"]),
            float(risk_factors["device_risk"]),
            float(risk_factors["browser_risk"]),
            float(risk_factors["time_risk"]),
            float(risk_factors["failed_attempt_risk"]),
            float(total_risk),
            float(not attempt.login_success)
        ] + [0.0] * 22  # Padding
        
        debug_to_file(f"Extracted features: {features}")
        debug_to_file(f"Feature length: {len(features)}")
        
        return features
    
    except Exception as e:
        debug_to_file(f"Feature extraction error: {e}")
        debug_to_file(traceback.format_exc())
        raise

def update_user_patterns(attempt):
    """
    Update user login patterns with new attempt and log details
    """
    try:
        debug_to_file(f"Updating patterns for user {attempt.user_id}")
        
        patterns = safe_load_json(PATTERNS_FILE)
        debug_to_file(f"Existing patterns before update: {patterns}")
        
        user_patterns = patterns.get(attempt.user_id, {
            "usual_locations": [],
            "usual_devices": [],
            "usual_times": [],
            "usual_browsers": [],
            "failed_attempts": 0,
            "total_attempts": 0,
            "last_successful_login": None
        })
        
        # Update patterns
        if attempt.location and attempt.location not in user_patterns.get("usual_locations", []):
            user_patterns.setdefault("usual_locations", []).append(attempt.location)
            debug_to_file(f"Added new location: {attempt.location}")
        
        if attempt.device_type not in user_patterns.get("usual_devices", []):
            user_patterns.setdefault("usual_devices", []).append(attempt.device_type)
            debug_to_file(f"Added new device type: {attempt.device_type}")
        
        if attempt.browser not in user_patterns.get("usual_browsers", []):
            user_patterns.setdefault("usual_browsers", []).append(attempt.browser)
            debug_to_file(f"Added new browser: {attempt.browser}")
        
        hour = datetime.fromtimestamp(attempt.timestamp).hour
        if hour not in user_patterns.get("usual_times", []):
            user_patterns.setdefault("usual_times", []).append(hour)
            debug_to_file(f"Added new usual time: {hour}")
        
        user_patterns["total_attempts"] = user_patterns.get("total_attempts", 0) + 1
        debug_to_file(f"Total attempts updated to: {user_patterns['total_attempts']}")
        
        if not attempt.login_success:
            user_patterns["failed_attempts"] = user_patterns.get("failed_attempts", 0) + 1
            debug_to_file(f"Failed attempts updated to: {user_patterns['failed_attempts']}")
        else:
            user_patterns["last_successful_login"] = attempt.timestamp
            debug_to_file(f"Last successful login updated to: {user_patterns['last_successful_login']}")
        
        patterns[attempt.user_id] = user_patterns
        safe_save_json(patterns, PATTERNS_FILE)
        
        debug_to_file(f"Updated patterns after save: {patterns}")
    
    except Exception as e:
        debug_to_file(f"Error updating user patterns: {str(e)}")
        debug_to_file(traceback.format_exc())
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint providing basic API information
    """
    return {
        "title": "Advanced Login Security API",
        "version": "1.1.0",
        "description": "API for detecting sophisticated login patterns and potential security threats",
        "status": "operational",
        "endpoints": [
            "/analyze_login (POST)",
            "/user_patterns/{user_id} (GET)"
        ]
    }

@app.post("/analyze_login", status_code=status.HTTP_200_OK)
async def analyze_login(attempt: LoginAttempt, response: Response):
    """
    Advanced login security analysis with comprehensive logging
    """
    try:
        debug_to_file(f"Analyzing login attempt for user {attempt.user_id}")
        debug_to_file(f"Full login attempt details: {attempt}")
        
        if model is None:
            debug_to_file("Security model is not available")
            raise HTTPException(status_code=503, detail="Security model not available")
        
        # Load existing patterns
        patterns = safe_load_json(PATTERNS_FILE)
        debug_to_file(f"Loaded patterns: {patterns}")
        
        # Extract features
        features = extract_features(attempt, patterns)
        
        # Validate features
        if len(features) != 30:
            debug_to_file(f"Invalid feature length. Expected 30, got {len(features)}")
            raise ValueError(f"Invalid feature length. Expected 30, got {len(features)}")
        
        # Get model prediction
        try:
            prediction = model.predict(np.array(features).reshape(1, -1))
            prediction_proba = getattr(model, 'predict_proba', lambda x: [[0.5, 0.5]])(np.array(features).reshape(1, -1))
            
            debug_to_file(f"Model prediction: {prediction}")
            debug_to_file(f"Model prediction probability: {prediction_proba}")
        except Exception as pred_error:
            debug_to_file(f"Prediction error: {str(pred_error)}")
            debug_to_file(traceback.format_exc())
            raise
        
        risk_score = float(prediction_proba[0][1])  # Probability of suspicious activity
        debug_to_file(f"Risk score: {risk_score}")
        
        # Update patterns with this attempt
        update_user_patterns(attempt)
        
        # Determine required actions based on risk score
        required_actions = []
        if risk_score > 0.8:
            required_actions.append("block_login")
            required_actions.append("notify_admin")
            required_actions.append("require_2fa")
            debug_to_file("High-risk login detected: blocking login and notifying admin")
        elif risk_score > 0.6:
            required_actions.append("require_2fa")
            required_actions.append("notify_user")
            debug_to_file("Moderate-risk login detected: requiring 2FA and notifying user")
        elif risk_score > 0.4:
            required_actions.append("monitor_closely")
            required_actions.append("challenge_questions")
            debug_to_file("Low-risk login detected: monitoring closely and challenging")
        
        debug_to_file(f"Required actions: {required_actions}")
        
        return {
            "risk_score": risk_score,
            "is_suspicious": bool(prediction[0]),
            "required_actions": required_actions,
            "risk_breakdown": {
                "location_risk": features[1],
                "device_risk": features[2],
                "browser_risk": features[3],
                "time_risk": features[4],
                "failed_attempt_risk": features[5]
            },
            "unusual_patterns": {
                "new_device": attempt.device_type not in patterns.get(attempt.user_id, {}).get("usual_devices", []),
                "new_location": attempt.location not in patterns.get(attempt.user_id, {}).get("usual_locations", []),
                "new_browser": attempt.browser not in patterns.get(attempt.user_id, {}).get("usual_browsers", []),
                "unusual_time": datetime.fromtimestamp(attempt.timestamp).hour not in patterns.get(attempt.user_id, {}).get("usual_times", [])
            }
        }
    except ValidationError as ve:
        debug_to_file(f"Validation error: {str(ve)}")
        debug_to_file(traceback.format_exc())
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        debug_to_file(f"Unexpected error in login analysis: {str(e)}")
        debug_to_file(traceback.format_exc())
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
