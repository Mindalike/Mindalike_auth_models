# Mindalike Models

This repository contains machine learning models for the **Mindalike** app, including:
- **Fraud Detection Model**: Detects suspicious login attempts.
- **Behavioral Biometrics Model**: Analyzes user behavior for additional security.
- **Login Security Model**: Advanced machine learning model for detecting suspicious login attempts using sophisticated feature engineering and ensemble learning techniques.

---

## **Table of Contents**
1. [Models Overview](#models-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Dataset](#dataset)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Deployment](#deployment)
8. [Contributing](#contributing)
9. [License](#license)

---

## **Models Overview**

### **1. Fraud Detection Model**
- **Algorithm**: Random Forest
- **Purpose**: Detects fraudulent login attempts based on features like IP address, login time, device type, and failed attempts.
- **Input Features**:
  - `ip_address`: IP address of the user.
  - `login_time`: Timestamp of the login attempt.
  - `device_type`: Type of device used (e.g., Mobile, Desktop).
  - `failed_attempts`: Number of failed login attempts.
- **Output**: `is_fraudulent` (1 for fraudulent, 0 for genuine).

### **2. Behavioral Biometrics Model**
- **Algorithm**: Recurrent Neural Network (RNN)
- **Purpose**: Analyzes user behavior (e.g., typing patterns, mouse movements) to detect anomalies.
- **Input Features**:
  - `typing_speed`: Typing speed in seconds.
  - `mouse_movement`: Mouse movement patterns.
  - `session_duration`: Duration of the user session.
- **Output**: `is_genuine` (1 for genuine, 0 for fraudulent).

### **3. Login Security Model**

#### Overview
Advanced machine learning model for detecting suspicious login attempts using sophisticated feature engineering and ensemble learning techniques.

#### Key Features
- Multi-model ensemble approach
- Advanced feature extraction
- Adaptive risk scoring
- Robust handling of class imbalance

#### Model Architecture
- **Ensemble Models**:
  1. XGBoost Classifier
  2. Random Forest Classifier
  3. Logistic Regression

#### Feature Engineering
### Time-Based Features
- Hour of day
- Day of week
- Login frequency
- Time since last login

### Risk-Based Features
- IP risk scoring
- Device entropy
- Browser entropy
- Location risk assessment

#### Performance Metrics
- **Target Accuracy**: 90%
- **Current Performance**:
  - ROC AUC Score: 0.85+
  - Average Precision: 0.80+

#### Deployment
- Deployed via FastAPI
- Supports real-time login risk assessment

#### Installation
```bash
pip install -r requirements.txt
```

#### Usage
```python
from train_model import LoginSecurityModel

# Load pre-trained model
model = LoginSecurityModel.load_model('login_security_model.joblib.gz')

# Predict login risk
result = model.predict(login_attempt_data)
```

#### Future Improvements
- Continuous model retraining
- Enhanced IP reputation integration
- More sophisticated anomaly detection

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mindalike-models.git
   cd mindalike-models
   
```

---

## **Usage**

### **Fraud Detection Model**
```python
from fraud_detection import FraudDetectionModel

# Load pre-trained model
model = FraudDetectionModel.load_model('fraud_detection_model.joblib.gz')

# Predict fraudulent login attempt
result = model.predict(login_attempt_data)
```

### **Behavioral Biometrics Model**
```python
from behavioral_biometrics import BehavioralBiometricsModel

# Load pre-trained model
model = BehavioralBiometricsModel.load_model('behavioral_biometrics_model.joblib.gz')

# Predict genuine user behavior
result = model.predict(user_behavior_data)
```

---

## **Dataset**

- **Fraud Detection Model**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Behavioral Biometrics Model**: [Custom dataset collected from user interactions]

---

## **Training**

- **Fraud Detection Model**: Trained using Random Forest Classifier with 100 trees and a maximum depth of 10.
- **Behavioral Biometrics Model**: Trained using Recurrent Neural Network (RNN) with 2 hidden layers and 128 units each.

---

## **Evaluation**

- **Fraud Detection Model**: Evaluated using ROC AUC Score and Average Precision.
- **Behavioral Biometrics Model**: Evaluated using Accuracy and F1 Score.

---

## **Deployment**

- **Fraud Detection Model**: Deployed via Flask API.
- **Behavioral Biometrics Model**: Deployed via FastAPI.

---

## **Contributing**

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

## **License**
[Specify your license here]