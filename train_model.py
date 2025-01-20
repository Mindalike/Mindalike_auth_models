import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import xgboost as xgb
from datetime import datetime, timedelta
import joblib
import gzip
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Set up logging
logging.basicConfig(
    filename='login_security_model.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LoginSecurityModel:
    def __init__(self, sensitivity_level=0.5):
        """
        Initialize login security model with configurable sensitivity
        """
        self.sensitivity_level = sensitivity_level
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.optimal_threshold = 0.5
    
    def _extract_time_features(self, df):
        """
        Extract advanced time-based features
        """
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='h')
        
        # Basic time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hour'] = df['hour'].between(9, 17).astype(int)
        df['is_late_night'] = df['hour'].between(0, 5).astype(int)
        
        # Time windows
        df['morning_window'] = df['hour'].between(6, 11).astype(int)
        df['afternoon_window'] = df['hour'].between(12, 17).astype(int)
        df['evening_window'] = df['hour'].between(18, 23).astype(int)
        
        return df
    
    def _extract_behavioral_features(self, df):
        """
        Extract behavioral patterns and risk features
        """
        # Group statistics
        for col in ['location', 'device_type', 'browser']:
            # Frequency encoding
            df[f'{col}_freq'] = df.groupby(col)[col].transform('count')
            df[f'{col}_freq_norm'] = df[f'{col}_freq'] / len(df)
            
            # Risk correlation
            df[f'{col}_avg_risk'] = df.groupby(col)['failed_attempt_risk'].transform('mean')
            df[f'{col}_risk_std'] = df.groupby(col)['failed_attempt_risk'].transform('std').fillna(0)
            
            # Time-based patterns
            df[f'{col}_hour_entropy'] = df.groupby(col)['hour'].transform(lambda x: -np.sum(np.unique(x, return_counts=True)[1] / len(x) * np.log2(np.unique(x, return_counts=True)[1] / len(x))))
            
            # Encode categorical variables
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                known_categories = set(self.label_encoders[col].classes_)
                df[col] = df[col].map(lambda x: list(known_categories)[0] if x not in known_categories else x)
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def _extract_risk_features(self, df):
        """
        Extract advanced risk-based features
        """
        risk_columns = [
            'location_risk', 'device_risk', 'browser_risk', 
            'time_risk', 'failed_attempt_risk'
        ]
        
        # Basic risk metrics
        df['composite_risk'] = df[risk_columns].mean(axis=1)
        df['max_risk'] = df[risk_columns].max(axis=1)
        df['min_risk'] = df[risk_columns].min(axis=1)
        df['risk_std'] = df[risk_columns].std(axis=1)
        df['risk_range'] = df['max_risk'] - df['min_risk']
        
        # Risk interactions
        df['location_device_risk'] = df['location_risk'] * df['device_risk']
        df['time_failed_attempt_risk'] = df['time_risk'] * df['failed_attempt_risk']
        df['browser_time_risk'] = df['browser_risk'] * df['time_risk']
        
        # Compound risk metrics
        df['geometric_mean_risk'] = df[risk_columns].apply(lambda x: np.exp(np.log(x + 1e-10).mean()), axis=1)
        df['harmonic_mean_risk'] = df[risk_columns].apply(lambda x: len(x) / np.sum(1 / (x + 1e-10)), axis=1)
        
        # Risk thresholds
        df['high_risk_count'] = df[risk_columns].apply(lambda x: np.sum(x > 0.7), axis=1)
        df['medium_risk_count'] = df[risk_columns].apply(lambda x: np.sum((x > 0.3) & (x <= 0.7)), axis=1)
        
        return df
    
    def preprocess_data(self, df):
        """
        Advanced preprocessing with enhanced feature engineering
        """
        df = df.copy()
        
        # Extract features
        df = self._extract_time_features(df)
        df = self._extract_behavioral_features(df)
        df = self._extract_risk_features(df)
        
        # Select features for model
        feature_columns = [
            # Time features
            'hour', 'day_of_week', 'is_weekend', 'is_business_hour', 'is_late_night',
            'morning_window', 'afternoon_window', 'evening_window',
            
            # Behavioral features
            'location_encoded', 'device_type_encoded', 'browser_encoded',
            'location_freq', 'device_type_freq', 'browser_freq',
            'location_freq_norm', 'device_type_freq_norm', 'browser_freq_norm',
            'location_avg_risk', 'device_type_avg_risk', 'browser_avg_risk',
            'location_risk_std', 'device_type_risk_std', 'browser_risk_std',
            'location_hour_entropy', 'device_type_hour_entropy', 'browser_hour_entropy',
            
            # Risk features
            'location_risk', 'device_risk', 'browser_risk', 
            'time_risk', 'failed_attempt_risk', 'composite_risk',
            'max_risk', 'min_risk', 'risk_std', 'risk_range',
            'location_device_risk', 'time_failed_attempt_risk', 'browser_time_risk',
            'geometric_mean_risk', 'harmonic_mean_risk',
            'high_risk_count', 'medium_risk_count'
        ]
        
        X = df[feature_columns]
        
        # Scale features
        if hasattr(self.scaler, 'mean_'):
            X = self.scaler.transform(X)
        else:
            X = self.scaler.fit_transform(X)
        
        return X, feature_columns
    
    def find_optimal_threshold(self, y_true, y_pred_proba):
        """
        Find optimal classification threshold using precision-recall curve
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx]
    
    def xgboost_cv_tuning(self, dtrain, params, num_boost_round=500):
        """
        Use XGBoost's cross-validation for hyperparameter tuning
        """
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            nfold=3,
            metrics={'error', 'auc'},
            early_stopping_rounds=20,
            seed=42
        )
        best_num_boost_round = len(cv_results)
        logging.info(f"Best num_boost_round: {best_num_boost_round}")
        return best_num_boost_round

    def train(self, training_data_path):
        """
        Train the login security model with advanced XGBoost configuration
        """
        try:
            # Load data
            logging.info("Loading training data...")
            df = pd.read_csv(training_data_path)
            
            # Preprocess data
            logging.info("Preprocessing data...")
            X, feature_columns = self.preprocess_data(df)
            y = df['login_success']
            
            # Handle class imbalance with SMOTE
            logging.info("Applying SMOTE to balance classes...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
            )
            
            # Convert to DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
            dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_columns)
            
            # Advanced XGBoost parameters - optimized for faster training
            params = {
                'objective': 'binary:logistic',
                'eval_metric': ['error', 'auc'],
                'max_depth': 6,
                'learning_rate': 0.1,  # Increased from 0.01
                'min_child_weight': 2,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'scale_pos_weight': 2,  # Increased to handle class imbalance
                'max_delta_step': 1,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
                'seed': 42
            }
            
            # Perform XGBoost CV
            logging.info("Performing XGBoost cross-validation...")
            best_num_boost_round = self.xgboost_cv_tuning(dtrain, params)
            
            # Train with early stopping
            logging.info("Training model...")
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=best_num_boost_round,
                evals=[(dtrain, 'train'), (dtest, 'test')],
                early_stopping_rounds=20,
                verbose_eval=50  # Reduced frequency of evaluation output
            )
            
            # Find optimal threshold
            y_pred_proba = self.model.predict(dtest)
            self.optimal_threshold = self.find_optimal_threshold(y_test, y_pred_proba)
            logging.info(f"Optimal threshold: {self.optimal_threshold:.4f}")
            
            # Make predictions with optimal threshold
            y_pred = (y_pred_proba > self.optimal_threshold).astype(int)
            
            # Print performance metrics
            print("\nModel Performance:")
            print(classification_report(y_test, y_pred))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"\nROC AUC Score: {roc_auc:.4f}")
            
            # Save model
            self.save_model()
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise
    
    def predict(self, features_df):
        """
        Make predictions with confidence scores
        """
        try:
            # Preprocess features
            X, feature_columns = self.preprocess_data(features_df.copy())
            dtest = xgb.DMatrix(X, feature_names=feature_columns)
            
            # Make prediction
            pred_proba = self.model.predict(dtest)[0]
            prediction = int(pred_proba > self.optimal_threshold)
            
            # Calculate confidence score
            confidence = abs(pred_proba - 0.5) * 2  # Scale to 0-1
            
            # Log prediction
            logging.info(f"Prediction: {prediction}, Probability: {pred_proba:.4f}, Confidence: {confidence:.4f}")
            
            return {
                'prediction': bool(prediction),
                'probability': float(pred_proba),
                'confidence': float(confidence),
                'needs_review': confidence < 0.4  # Flag for review if confidence is low
            }
            
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            raise
    
    def save_model(self, filename='login_security_model.joblib.gz'):
        """
        Save trained model with compression
        """
        with gzip.open(filename, 'wb') as f:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'sensitivity_level': self.sensitivity_level,
                'optimal_threshold': self.optimal_threshold
            }, f)
        logging.info(f"Model saved to {filename}")
    
    @classmethod
    def load_model(cls, filename='login_security_model.joblib.gz'):
        """
        Load pre-trained model
        """
        with gzip.open(filename, 'rb') as f:
            model_data = joblib.load(f)
        
        loaded_model = cls(model_data['sensitivity_level'])
        loaded_model.model = model_data['model']
        loaded_model.scaler = model_data['scaler']
        loaded_model.label_encoders = model_data['label_encoders']
        loaded_model.optimal_threshold = model_data['optimal_threshold']
        
        return loaded_model

def main():
    # Train model with default sensitivity
    model = LoginSecurityModel(sensitivity_level=0.5)
    model.train('login_attempts_training_data.csv')

if __name__ == "__main__":
    main()
