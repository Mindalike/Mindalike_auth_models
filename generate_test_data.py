import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from scipy.stats import entropy

# Initialize Faker
fake = Faker()

def calculate_entropy(series):
    """Calculate entropy of a series"""
    value_counts = series.value_counts(normalize=True)
    return entropy(value_counts)

def generate_test_data(num_samples=10000, fraud_ratio=0.5):
    """Generate synthetic test data for login security model"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Lists to store data
    data = []
    
    # Load training data to get the same categorical values
    train_df = pd.read_csv('login_attempts_training_data.csv')
    user_ids = train_df['user_id'].unique().tolist()
    locations = train_df['location'].unique().tolist()
    device_types = train_df['device_type'].unique().tolist()
    browsers = train_df['browser'].unique().tolist()
    ip_addresses = train_df['ip_address'].unique().tolist()
    
    # Generate base timestamp
    base_timestamp = datetime.now() - timedelta(days=30)
    
    for _ in range(num_samples):
        # Randomly decide if this will be a fraudulent attempt
        is_fraud = random.random() < fraud_ratio
        
        # Generate timestamp
        timestamp = int((base_timestamp + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )).timestamp())
        
        # Generate user behavior features
        user_id = random.choice(user_ids)
        location = random.choice(locations)
        device_type = random.choice(device_types)
        browser = random.choice(browsers)
        ip_address = random.choice(ip_addresses)
        
        # Generate risk scores based on fraud status
        if is_fraud:
            base_risk = random.uniform(0.6, 1.0)
            login_success = 0
        else:
            base_risk = random.uniform(0.0, 0.4)
            login_success = 1
        
        # Add some randomness to risk scores
        location_risk = base_risk + random.uniform(-0.1, 0.1)
        device_risk = base_risk + random.uniform(-0.1, 0.1)
        browser_risk = base_risk + random.uniform(-0.1, 0.1)
        time_risk = base_risk + random.uniform(-0.1, 0.1)
        ip_risk_score = base_risk + random.uniform(-0.1, 0.1)
        
        # Ensure risk scores are between 0 and 1
        location_risk = max(0, min(1, location_risk))
        device_risk = max(0, min(1, device_risk))
        browser_risk = max(0, min(1, browser_risk))
        time_risk = max(0, min(1, time_risk))
        ip_risk_score = max(0, min(1, ip_risk_score))
        
        # Generate login frequency and time since last login
        login_frequency = random.randint(1, 100)
        time_since_last_login = random.randint(300, 86400)  # 5 minutes to 24 hours in seconds
        
        # Calculate failed attempt risk
        failed_attempts = random.randint(0, 5) if is_fraud else random.randint(0, 2)
        failed_attempt_risk = min(1.0, failed_attempts / 5)
        
        # Generate entry
        entry = {
            'timestamp': timestamp,
            'user_id': user_id,
            'location': location,
            'device_type': device_type,
            'browser': browser,
            'ip_address': ip_address,
            'login_frequency': login_frequency,
            'time_since_last_login': time_since_last_login,
            'location_risk': location_risk,
            'device_risk': device_risk,
            'browser_risk': browser_risk,
            'time_risk': time_risk,
            'failed_attempt_risk': failed_attempt_risk,
            'ip_risk_score': ip_risk_score,
            'login_success': login_success
        }
        
        data.append(entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate entropy features
    df['device_entropy'] = df.groupby('user_id')['device_type'].transform(calculate_entropy)
    df['browser_entropy'] = df.groupby('user_id')['browser'].transform(calculate_entropy)
    df['location_entropy'] = df.groupby('user_id')['location'].transform(calculate_entropy)
    df['ip_entropy'] = df.groupby('user_id')['ip_address'].transform(calculate_entropy)
    
    # Save to CSV
    df.to_csv('login_attempts_test_data.csv', index=False)
    print(f"Generated {num_samples} test samples with {fraud_ratio*100}% fraud ratio")
    print(f"Data saved to login_attempts_test_data.csv")
    
    return df

if __name__ == "__main__":
    generate_test_data()
