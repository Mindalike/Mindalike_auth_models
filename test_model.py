import pandas as pd
import numpy as np
import joblib
import json
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)

def load_model_artifacts():
    """Load the trained model and its artifacts"""
    try:
        model = joblib.load('model_artifacts/model.joblib')
        scaler = joblib.load('model_artifacts/scaler.joblib')
        label_encoders = joblib.load('model_artifacts/label_encoders.joblib')
        
        with open('model_artifacts/feature_columns.json', 'r') as f:
            feature_cols = json.load(f)
            
        return model, scaler, label_encoders, feature_cols
    except Exception as e:
        logging.error(f"Error loading model artifacts: {str(e)}")
        raise

def preprocess_test_data(df, scaler, label_encoders, feature_cols):
    """Preprocess test data using the same transformations as training"""
    try:
        df = df.copy()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Advanced time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_late_night'] = ((df['hour'] >= 23) | (df['hour'] <= 4)).astype(int)
        
        # User behavior features
        df['user_activity_count'] = df.groupby('user_id')['timestamp'].transform('count')
        df['user_success_rate'] = df.groupby('user_id')['login_success'].transform('mean')
        
        # Location-based features
        df['location_success_rate'] = df.groupby('location')['login_success'].transform('mean')
        df['location_activity_count'] = df.groupby('location')['timestamp'].transform('count')
        
        # Device-based features
        df['device_success_rate'] = df.groupby('device_type')['login_success'].transform('mean')
        df['device_activity_count'] = df.groupby('device_type')['timestamp'].transform('count')
        
        # Browser-based features
        df['browser_success_rate'] = df.groupby('browser')['login_success'].transform('mean')
        df['browser_activity_count'] = df.groupby('browser')['timestamp'].transform('count')
        
        # Time window features
        df['recent_failures'] = df.groupby('user_id')['login_success'].transform(
            lambda x: (~x.rolling(window=3, min_periods=1).sum().astype(bool)).astype(int)
        )
        
        # Interaction features
        df['location_device_risk'] = df.groupby(['location', 'device_type'])['login_success'].transform('mean')
        df['browser_time_risk'] = df.groupby(['browser', 'hour'])['login_success'].transform('mean')
        
        # Handle categorical columns
        for col, encoder in label_encoders.items():
            if col in df.columns and col not in ['timestamp', 'login_success']:
                df[col] = encoder.transform(df[col].fillna('unknown'))
        
        # Handle missing values
        for col in feature_cols:
            if col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(-1)
        
        # Select and scale features
        X = df[feature_cols]
        X_scaled = scaler.transform(X)
        
        return X_scaled
    
    except Exception as e:
        logging.error(f"Error preprocessing test data: {str(e)}")
        raise

def evaluate_model(y_true, y_pred, y_pred_proba):
    """Calculate and log model performance metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }
    
    logging.info("\nModel Performance on Test Data:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_true, y_pred))
    
    return metrics

def test_model(test_data_path):
    """Test the trained model on new data"""
    try:
        # Load model artifacts
        logging.info("Loading model artifacts...")
        model, scaler, label_encoders, feature_cols = load_model_artifacts()
        
        # Load and preprocess test data
        logging.info("Loading test data...")
        test_df = pd.read_csv(test_data_path)
        
        logging.info("Preprocessing test data...")
        X_test = preprocess_test_data(test_df, scaler, label_encoders, feature_cols)
        y_test = test_df['login_success'].values
        
        # Make predictions
        logging.info("Making predictions...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate performance
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Save test metrics
        with open('model_artifacts/test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logging.info("\nTest metrics saved to model_artifacts/test_metrics.json")
        
        return metrics
    
    except Exception as e:
        logging.error(f"Error testing model: {str(e)}")
        raise

if __name__ == "__main__":
    test_model('login_attempts_test_data.csv')
