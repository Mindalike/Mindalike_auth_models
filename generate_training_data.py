import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
fake = Faker()

def generate_login_data(num_samples=10000):
    # Prepare lists to store data
    data = {
        'timestamp': [],
        'user_id': [],
        'ip_address': [],
        'location': [],
        'device_type': [],
        'browser': [],
        'login_success': []
    }
    
    # Predefined lists for more realistic data
    locations = ['New York', 'London', 'Tokyo', 'Sydney', 'Paris', 'Berlin', 'Unknown']
    device_types = ['desktop', 'mobile', 'tablet']
    browsers = ['Chrome', 'Firefox', 'Safari', 'Edge']
    
    # Generate synthetic data
    for _ in range(num_samples):
        # Timestamp: Distributed across recent times
        timestamp = datetime.now() - timedelta(days=random.randint(0, 365))
        
        # User ID
        user_id = f'user_{random.randint(1, 1000)}'
        
        # IP Address
        ip_address = f'{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}'
        
        # Location
        location = random.choice(locations)
        
        # Device and Browser
        device_type = random.choice(device_types)
        browser = random.choice(browsers)
        
        # Login Success (with slight imbalance)
        login_success = 1 if random.random() > 0.3 else 0
        
        # Store data
        data['timestamp'].append(timestamp.timestamp())
        data['user_id'].append(user_id)
        data['ip_address'].append(ip_address)
        data['location'].append(location)
        data['device_type'].append(device_type)
        data['browser'].append(browser)
        data['login_success'].append(login_success)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add risk features
    df['location_risk'] = df['location'].apply(lambda x: 0.8 if x == 'Unknown' else 0.2)
    df['device_risk'] = df['device_type'].apply(lambda x: 0.6 if x == 'mobile' else 0.2)
    df['browser_risk'] = df['browser'].apply(lambda x: 0.5 if x == 'Edge' else 0.1)
    df['time_risk'] = df['timestamp'].apply(lambda x: 0.7 if datetime.fromtimestamp(x).hour in [0, 23] else 0.2)
    df['failed_attempt_risk'] = df['login_success'].apply(lambda x: 0.8 if x == 0 else 0.1)
    
    # Total risk calculation
    df['total_risk'] = (df['location_risk'] + df['device_risk'] + 
                        df['browser_risk'] + df['time_risk'] + 
                        df['failed_attempt_risk']) / 5
    
    return df

# Generate and save dataset
df = generate_login_data()
df.to_csv('login_attempts_training_data.csv', index=False)
print(f"Generated {len(df)} synthetic login attempts")
print("Dataset saved to login_attempts_training_data.csv")
