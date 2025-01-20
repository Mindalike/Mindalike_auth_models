import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random

class LoginDataGenerator:
    def __init__(self, num_samples=50000, random_seed=42):
        """
        Initialize synthetic login data generator
        
        Args:
            num_samples (int): Number of synthetic login attempts to generate
            random_seed (int): Random seed for reproducibility
        """
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.fake = Faker()
        self.num_samples = num_samples
        
        # Predefined risk categories
        self.risk_categories = {
            'low_risk': (0.1, 0.3),
            'medium_risk': (0.3, 0.7),
            'high_risk': (0.7, 1.0)
        }
        
        # Device and browser configurations
        self.devices = ['desktop', 'mobile', 'tablet', 'smart_tv']
        self.browsers = ['chrome', 'firefox', 'safari', 'edge', 'opera']
        self.locations = ['US', 'EU', 'APAC', 'LATAM', 'MENA']
    
    def _generate_timestamp(self):
        """
        Generate realistic timestamps with time zone variations
        """
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        
        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + timedelta(days=random_number_of_days)
        
        # Add random time of day
        random_time = timedelta(
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        timestamp = random_date + random_time
        return int(timestamp.timestamp())
    
    def _calculate_login_risk(self, features):
        """
        Calculate comprehensive login risk score
        
        Args:
            features (dict): Login attempt features
        
        Returns:
            float: Calculated risk score
        """
        # Base risk calculation
        base_risk = 0.5
        
        # Time-based risk
        hour = datetime.fromtimestamp(features['timestamp']).hour
        if hour < 6 or hour > 22:
            base_risk += 0.2  # Higher risk during odd hours
        
        # Device risk
        device_risk_map = {
            'desktop': 0.3,
            'mobile': 0.5,
            'tablet': 0.4,
            'smart_tv': 0.6
        }
        base_risk += device_risk_map.get(features['device_type'], 0.5)
        
        # Location risk
        location_risk_map = {
            'US': 0.3,
            'EU': 0.4,
            'APAC': 0.5,
            'LATAM': 0.6,
            'MENA': 0.7
        }
        base_risk += location_risk_map.get(features['location'], 0.5)
        
        # Browser risk
        browser_risk_map = {
            'chrome': 0.3,
            'firefox': 0.4,
            'safari': 0.5,
            'edge': 0.4,
            'opera': 0.6
        }
        base_risk += browser_risk_map.get(features['browser'], 0.5)
        
        # Normalize risk
        return max(0, min(1, base_risk))
    
    def generate_login_attempts(self):
        """
        Generate synthetic login attempts with comprehensive features
        
        Returns:
            pd.DataFrame: DataFrame of synthetic login attempts
        """
        login_attempts = []
        
        # Balanced dataset parameters
        total_samples = self.num_samples
        successful_login_ratio = 0.5  # 50% successful, 50% failed
        
        successful_samples = int(total_samples * successful_login_ratio)
        failed_samples = total_samples - successful_samples
        
        for is_successful in [1, 0]:
            current_samples = successful_samples if is_successful else failed_samples
            
            for _ in range(current_samples):
                # Basic login features
                timestamp = self._generate_timestamp()
                device_type = random.choice(self.devices)
                browser = random.choice(self.browsers)
                location = random.choice(self.locations)
                
                # Advanced login features
                login_attempt = {
                    'timestamp': timestamp,
                    'user_id': self.fake.uuid4(),
                    'device_type': device_type,
                    'browser': browser,
                    'location': location,
                    'ip_address': self.fake.ipv4(),
                    
                    # Entropy and complexity features
                    'device_entropy': random.random(),
                    'browser_entropy': random.random(),
                    'location_entropy': random.random(),
                    
                    # Login pattern features
                    'login_frequency': random.randint(1, 100),
                    'time_since_last_login': random.randint(0, 86400),  # Seconds
                }
                
                # Calculate risk scores
                login_attempt['location_risk'] = random.uniform(0, 1)
                login_attempt['device_risk'] = random.uniform(0, 1)
                login_attempt['browser_risk'] = random.uniform(0, 1)
                login_attempt['time_risk'] = random.uniform(0, 1)
                login_attempt['failed_attempt_risk'] = random.uniform(0, 1)
                login_attempt['ip_risk_score'] = random.uniform(0, 1)
                
                # Explicitly set login success
                login_attempt['login_success'] = is_successful
                
                login_attempts.append(login_attempt)
        
        return pd.DataFrame(login_attempts)

def main():
    """
    Generate and save synthetic login attempts dataset
    """
    generator = LoginDataGenerator(num_samples=50000)
    dataset = generator.generate_login_attempts()
    
    # Save dataset
    dataset.to_csv('login_attempts_training_data.csv', index=False)
    print(f"Generated {len(dataset)} synthetic login attempts")
    print("Dataset saved to login_attempts_training_data.csv")

if __name__ == "__main__":
    main()
