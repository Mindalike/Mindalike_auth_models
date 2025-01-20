import pandas as pd
import numpy as np
import logging
import traceback
import gzip
import joblib
import json
import os
from datetime import datetime, timedelta

# Advanced ML Libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

class LoginSecurityModel:
    def __init__(self, sensitivity_level=0.5):
        """
        Initialize login security model with advanced configuration
        """
        self.sensitivity_level = sensitivity_level
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.optimal_threshold = 0.5
        self.feature_importances = {}
        
    def _extract_advanced_features(self, df):
        """
        Extract comprehensive and advanced features for login security
        """
        # Time-based features
        df['hour_of_day'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp'], unit='s').dt.dayofweek
        
        # IP-based risk features
        df['ip_risk_score'] = self._calculate_ip_risk(df['ip_address'])
        
        # Device and browser entropy
        df['device_entropy'] = df.groupby('device_type')['user_id'].transform('count')
        df['browser_entropy'] = df.groupby('browser')['user_id'].transform('count')
        
        # Location-based features
        df['location_risk_score'] = self._calculate_location_risk(df['location'])
        
        # Temporal login patterns
        df['login_frequency'] = df.groupby('user_id')['timestamp'].transform('count')
        df['time_since_last_login'] = df.groupby('user_id')['timestamp'].transform(lambda x: x.max() - x.min())
        
        return df
    
    def _calculate_ip_risk(self, ip_addresses):
        """
        Calculate risk score based on IP characteristics
        """
        # Simple IP risk scoring (can be enhanced with IP reputation databases)
        def ip_to_risk(ip):
            # Example risk calculation logic
            octets = ip.split('.')
            risk = sum(int(octet) % 10 for octet in octets) / 40
            return min(risk, 1.0)
        
        return [ip_to_risk(ip) for ip in ip_addresses]
    
    def _calculate_location_risk(self, locations):
        """
        Calculate location risk based on historical patterns
        """
        # Simple location risk scoring
        def location_risk(location):
            # Example: Define high-risk locations
            high_risk_locations = ['Unknown', 'Tor Exit Node', 'VPN']
            return 1.0 if location in high_risk_locations else 0.2
        
        return [location_risk(loc) for loc in locations]
    
    def preprocess_data(self, df):
        """
        Advanced data preprocessing with feature engineering
        """
        # Add advanced features
        df = self._extract_advanced_features(df)
        
        # Encode categorical features
        categorical_columns = ['device_type', 'browser', 'location']
        for col in categorical_columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Select and scale features
        feature_columns = [
            'timestamp', 'hour_of_day', 'day_of_week', 
            'ip_risk_score', 'device_entropy', 'browser_entropy', 
            'location_risk_score', 'login_frequency', 'time_since_last_login',
            'device_type_encoded', 'browser_encoded', 'location_encoded'
        ]
        
        X = df[feature_columns]
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, feature_columns
    
    def _create_advanced_model(self):
        """
        Create an advanced model
        """
        # Base models
        xgb_model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10
        )
        
        return xgb_model  # Simplified to single model
    
    def train(self, training_data_path, test_size=0.2, random_state=42):
        """
        Advanced model training with comprehensive techniques
        """
        try:
            # Load data
            logging.info("Loading training data...")
            df = pd.read_csv(training_data_path)
            
            # Preprocess data
            logging.info("Preprocessing data...")
            X, feature_columns = self.preprocess_data(df)
            y = df['login_success']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Create advanced model
            logging.info("Creating advanced ensemble model...")
            self.model = self._create_advanced_model()
            
            # Train model
            logging.info("Training model...")
            self.model.fit(X_train, y_train)
            
            # Predictions
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Find optimal threshold
            self.optimal_threshold = self.find_optimal_threshold(y_test, y_pred_proba)
            y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)
            
            # Detailed performance metrics
            print("\nModel Performance:")
            print(classification_report(y_test, y_pred))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            
            # Calculate advanced metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            print(f"\nROC AUC Score: {roc_auc:.4f}")
            print(f"Average Precision Score: {avg_precision:.4f}")
            
            # Feature importance analysis
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_importances = sorted(
                    zip(feature_columns, importances), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                print("\nTop Feature Importances:")
                for feature, importance in feature_importances[:10]:
                    print(f"{feature}: {importance:.4f}")
                
                self.feature_importances = dict(feature_importances)
            
            # Save model
            self.save_model()
            
            return self.model
        
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def find_optimal_threshold(self, y_true, y_pred_proba, beta=1.0):
        """
        Find optimal threshold using F-beta score
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls)
        optimal_idx = np.argmax(f_scores)
        return thresholds[optimal_idx]
    
    def save_model(self, filename='login_security_model.joblib.gz'):
        """
        Save model with advanced compression
        """
        try:
            with gzip.open(filename, 'wb') as f:
                joblib.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'label_encoders': self.label_encoders,
                    'optimal_threshold': self.optimal_threshold,
                    'feature_importances': self.feature_importances
                }, f, compress=9)
            logging.info(f"Model saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
    
    def load_model(self, filename='login_security_model.joblib.gz'):
        """
        Load model with robust error handling
        """
        try:
            with gzip.open(filename, 'rb') as f:
                model_data = joblib.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.label_encoders = model_data['label_encoders']
                self.optimal_threshold = model_data.get('optimal_threshold', 0.5)
                self.feature_importances = model_data.get('feature_importances', {})
            logging.info(f"Model loaded from {filename}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

def main():
    model = LoginSecurityModel()
    model.train('login_attempts_training_data.csv')

if __name__ == "__main__":
    main()
