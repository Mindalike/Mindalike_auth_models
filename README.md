# Mindalike Models

This repository contains machine learning models for the **Mindalike** app, including:
- **Fraud Detection Model**: Detects suspicious login attempts.
- **Behavioral Biometrics Model**: Analyzes user behavior for additional security.

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

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mindalike-models.git
   cd mindalike-models
