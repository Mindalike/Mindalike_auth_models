import pandas as pd
import numpy as np
import logging
import json
import os
import traceback
import joblib
from datetime import datetime
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    roc_auc_score,
    precision_recall_curve, 
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
from scipy import stats
import xgboost as xgb
from imblearn.over_sampling import BorderlineSMOTE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)

def preprocess_data(df, scaler=None, label_encoders=None):
    """
    Enhanced data preprocessing with advanced feature engineering
    """
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
        
        # Handle categorical columns with advanced encoding
        categorical_cols = df.select_dtypes(include=['object']).columns
        if label_encoders is None:
            label_encoders = {}
        
        for col in categorical_cols:
            if col not in ['timestamp']:
                if col not in label_encoders:
                    label_encoders[col] = LabelEncoder()
                    df[col] = label_encoders[col].fit_transform(df[col].fillna('unknown'))
                else:
                    df[col] = label_encoders[col].transform(df[col].fillna('unknown'))
        
        # Select features for model
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'login_success']]
        
        # Handle missing values
        for col in feature_cols:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(-1)
        
        # Scale features
        if scaler is None:
            scaler = StandardScaler()
            X = scaler.fit_transform(df[feature_cols])
        else:
            X = scaler.transform(df[feature_cols])
        
        return X, feature_cols, scaler, label_encoders
    
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        raise

def train_model(data_path, test_size=0.2, random_state=42):
    """
    Train an enhanced login security model
    """
    try:
        # Load data
        logging.info("Loading data...")
        df = pd.read_csv(data_path)
        
        # Preprocess data
        logging.info("Preprocessing data...")
        X, feature_cols, scaler, label_encoders = preprocess_data(df)
        y = df['login_success'].values
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Handle class imbalance with advanced SMOTE
        logging.info("Applying SMOTE...")
        smote = BorderlineSMOTE(
            random_state=random_state,
            k_neighbors=5,
            m_neighbors=10,
            kind='borderline-1'
        )
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Initialize model with optimized parameters
        logging.info("Training model...")
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=1,
            random_state=random_state,
            tree_method='hist',
            enable_categorical=True
        )
        
        # Train the model
        model.fit(
            X_train_balanced, 
            y_train_balanced,
            verbose=True
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Log metrics
        logging.info("\nModel Performance:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logging.info(f"{metric}: {value:.4f}")
        logging.info(f"\nClassification Report:\n{metrics['classification_report']}")
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        logging.info("\nTop 10 Most Important Features:")
        logging.info(feature_importance.head(10))
        
        # Save artifacts
        os.makedirs('model_artifacts', exist_ok=True)
        joblib.dump(model, 'model_artifacts/model.joblib')
        joblib.dump(scaler, 'model_artifacts/scaler.joblib')
        joblib.dump(label_encoders, 'model_artifacts/label_encoders.joblib')
        
        with open('model_artifacts/feature_columns.json', 'w') as f:
            json.dump(feature_cols, f)
        
        with open('model_artifacts/metrics.json', 'w') as f:
            metrics_json = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            json.dump(metrics_json, f, indent=4)
        
        # Save feature importance
        feature_importance.to_csv('model_artifacts/feature_importance.csv', index=False)
        
        return model, scaler, label_encoders, metrics
    
    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    train_model('login_attempts_training_data.csv')
