"""
AutoML engine using scikit-learn and other ML libraries.
Handles data preprocessing, model training, and evaluation.
"""

import pandas as pd
import numpy as np
import pickle
import io
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import streamlit as st


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Preprocess dataset and detect problem type.
    Returns cleaned dataframe and problem type ('classification' or 'regression').
    """
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Handle missing values
    missing_percent = df_clean.isnull().sum().sum() / (df_clean.shape[0] * df_clean.shape[1]) * 100
    
    if missing_percent > 50:
        st.warning(f"Dataset has {missing_percent:.1f}% missing values. Consider cleaning your data.")
    
    # Fill missing values
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Detect problem type based on target column
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.error("No numeric columns found. Please ensure your dataset has numeric features.")
        return df_clean, 'classification'
    
    # For now, assume classification if target has few unique values
    # This will be refined when user selects target column
    return df_clean, 'classification'


def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Detect if the problem is classification or regression based on target column.
    """
    unique_values = df[target_col].nunique()
    total_values = len(df[target_col])
    
    # If target has few unique values relative to dataset size, likely classification
    if unique_values <= 20 or unique_values / total_values < 0.1:
        return 'classification'
    else:
        return 'regression'


def train_model(df: pd.DataFrame, target_col: str, problem_type: str, test_split: float = 0.2) -> Dict[str, Any]:
    """
    Train AutoML model using scikit-learn and other ML libraries.
    Returns dictionary with model, metrics, and feature importance.
    """
    try:
        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models to try
        if problem_type == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
                'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
            }
        else:
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
                'XGBoost': xgb.XGBRegressor(random_state=42),
                'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1)
            }
        
        # Train and evaluate models
        best_model = None
        best_score = -np.inf
        best_model_name = ""
        model_scores = {}
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                if problem_type == 'classification':
                    y_pred = model.predict(X_test_scaled)
                    score = accuracy_score(y_test, y_pred)
                else:
                    y_pred = model.predict(X_test_scaled)
                    score = r2_score(y_test, y_pred)
                
                model_scores[name] = score
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                st.warning(f"Failed to train {name}: {str(e)}")
                continue
        
        if best_model is None:
            st.error("No models could be trained successfully.")
            return None
        
        # Calculate detailed metrics
        y_pred = best_model.predict(X_test_scaled)
        
        if problem_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
        else:
            metrics = {
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': np.mean(np.abs(y_test - y_pred))
            }
        
        # Get feature importance
        feature_importance = get_feature_importance(best_model, X.columns)
        
        return {
            'model': best_model,
            'model_name': best_model_name,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'model_scores': model_scores
        }
        
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None


def get_feature_importance(model, feature_names) -> Dict[str, float]:
    """
    Extract feature importance from trained model.
    Returns dictionary of feature names and importance scores.
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            coef = np.abs(model.coef_)
            if coef.ndim > 1:
                coef = coef[0]  # Take first class for multi-class
            importance_dict = dict(zip(feature_names, coef))
        else:
            return {}
        
        # Return top 10 features
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:10])
        
    except Exception as e:
        st.warning(f"Could not extract feature importance: {str(e)}")
        return {}


def generate_predictions_sample(model, scaler, label_encoders, test_data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Generate sample predictions for visualization.
    Returns dataframe with actual vs predicted values.
    """
    try:
        # Prepare test data
        X_test = test_data.drop(columns=[target_col])
        
        # Apply label encoding
        for col, le in label_encoders.items():
            if col in X_test.columns:
                X_test[col] = le.transform(X_test[col].astype(str))
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Generate predictions
        predictions = model.predict(X_test_scaled)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'actual': test_data[target_col].values,
            'predicted': predictions
        })
        
        # Return first 20 rows for visualization
        return comparison_df.head(20)
        
    except Exception as e:
        st.warning(f"Could not generate predictions: {str(e)}")
        return pd.DataFrame()


def serialize_model(model) -> bytes:
    """
    Serialize model to bytes for storage.
    """
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    return buffer.getvalue()


def deserialize_model(model_bytes: bytes):
    """
    Deserialize model from bytes.
    """
    buffer = io.BytesIO(model_bytes)
    return pickle.load(buffer)
