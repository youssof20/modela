"""
PyCaret AutoML wrapper functions for model training and evaluation.
Handles data preprocessing, model training, and evaluation.
"""

import pandas as pd
import numpy as np
import pickle
import io
from typing import Dict, Any, Tuple, Optional
from pycaret.classification import setup as clf_setup, compare_models as clf_compare, pull, create_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare
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
    Train AutoML model using PyCaret.
    Returns dictionary with model, metrics, and feature importance.
    """
    try:
        # Set up PyCaret environment
        if problem_type == 'classification':
            setup = clf_setup(
                data=df,
                target=target_col,
                test_data=None,
                preprocess=True,
                session_id=42,
                train_size=1-test_split,
                fold=3,  # Fast validation
                silent=True,
                verbose=False
            )
            
            # Compare models (limit to 5 for speed)
            best_model = clf_compare(
                include=['lr', 'rf', 'xgboost', 'lightgbm', 'catboost'],
                n_select=1,
                sort='Accuracy',
                verbose=False
            )
            
        else:  # regression
            setup = reg_setup(
                data=df,
                target=target_col,
                test_data=None,
                preprocess=True,
                session_id=42,
                train_size=1-test_split,
                fold=3,  # Fast validation
                silent=True,
                verbose=False
            )
            
            # Compare models (limit to 5 for speed)
            best_model = reg_compare(
                include=['lr', 'rf', 'xgboost', 'lightgbm', 'catboost'],
                n_select=1,
                sort='RMSE',
                verbose=False
            )
        
        # Get model results
        results = pull()
        
        # Get feature importance
        feature_importance = get_feature_importance(best_model[0])
        
        # Get model name
        model_name = str(best_model[0]).split('(')[0]
        
        # Prepare metrics
        metrics = {}
        if problem_type == 'classification':
            metrics = {
                'accuracy': results.loc[0, 'Accuracy'],
                'precision': results.loc[0, 'Precision'],
                'recall': results.loc[0, 'Recall'],
                'f1': results.loc[0, 'F1'],
                'auc': results.loc[0, 'AUC']
            }
        else:
            metrics = {
                'rmse': results.loc[0, 'RMSE'],
                'mae': results.loc[0, 'MAE'],
                'r2': results.loc[0, 'R2'],
                'mape': results.loc[0, 'MAPE']
            }
        
        return {
            'model': best_model[0],
            'model_name': model_name,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'results_df': results
        }
        
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None


def get_feature_importance(model) -> Dict[str, float]:
    """
    Extract feature importance from trained model.
    Returns dictionary of feature names and importance scores.
    """
    try:
        # Get feature importance from PyCaret
        importance_df = model.feature_importance()
        
        # Convert to dictionary and sort by importance
        importance_dict = dict(zip(importance_df['Feature'], importance_df['Importance']))
        
        # Return top 10 features
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:10])
        
    except Exception as e:
        st.warning(f"Could not extract feature importance: {str(e)}")
        return {}


def generate_predictions_sample(model, test_data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Generate sample predictions for visualization.
    Returns dataframe with actual vs predicted values.
    """
    try:
        # Generate predictions
        predictions = model.predict(test_data.drop(columns=[target_col]))
        
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


def get_model_explanations(model, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Get model explanations and insights.
    Returns dictionary with various model insights.
    """
    try:
        explanations = {}
        
        # Get feature importance
        explanations['feature_importance'] = get_feature_importance(model)
        
        # Get model performance summary
        results = pull()
        explanations['performance_summary'] = results.iloc[0].to_dict()
        
        # Get data summary
        explanations['data_summary'] = {
            'total_samples': len(df),
            'total_features': len(df.columns) - 1,
            'target_column': target_col,
            'missing_values': df.isnull().sum().sum()
        }
        
        return explanations
        
    except Exception as e:
        st.warning(f"Could not generate model explanations: {str(e)}")
        return {}
