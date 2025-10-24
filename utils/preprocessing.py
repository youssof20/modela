"""
Data preprocessing and validation utilities.
Handles dataset validation, cleaning, and preparation for training.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import streamlit as st


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate dataset for AutoML training.
    Returns (is_valid, list_of_warnings).
    """
    warnings = []
    
    # Check dataset size
    if len(df) > 100000:
        warnings.append("Dataset has more than 100,000 rows. Training may take longer.")
    
    if len(df) < 10:
        return False, ["Dataset must have at least 10 rows for training."]
    
    if len(df.columns) < 2:
        return False, ["Dataset must have at least 2 columns (features + target)."]
    
    # Check for missing values
    missing_percent = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    if missing_percent > 50:
        warnings.append(f"Dataset has {missing_percent:.1f}% missing values. Consider cleaning your data.")
    
    # Check for duplicate rows
    duplicate_percent = df.duplicated().sum() / len(df) * 100
    if duplicate_percent > 20:
        warnings.append(f"Dataset has {duplicate_percent:.1f}% duplicate rows. Consider removing duplicates.")
    
    # Check for columns with all same values
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        warnings.append(f"Columns with constant values detected: {', '.join(constant_cols)}")
    
    return True, warnings


def suggest_target_column(df: pd.DataFrame) -> str:
    """
    Suggest the best target column based on data characteristics.
    Returns column name.
    """
    # Priority order for target column suggestions
    priority_keywords = ['target', 'label', 'class', 'outcome', 'result', 'prediction', 'y']
    
    # Check for columns with priority keywords
    for keyword in priority_keywords:
        for col in df.columns:
            if keyword.lower() in col.lower():
                return col
    
    # If no priority keywords found, suggest based on data type
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # For classification, prefer categorical columns
    if categorical_cols:
        # Choose categorical column with reasonable number of unique values
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 20:
                return col
    
    # For regression, prefer numeric columns
    if numeric_cols:
        return numeric_cols[0]
    
    # Fallback to first column
    return df.columns[0]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset by handling missing values and data types.
    Returns cleaned dataframe.
    """
    df_clean = df.copy()
    
    # Handle missing values
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # For categorical columns, fill with mode or 'Unknown'
            mode_value = df_clean[col].mode()
            if not mode_value.empty:
                df_clean[col] = df_clean[col].fillna(mode_value[0])
            else:
                df_clean[col] = df_clean[col].fillna('Unknown')
        else:
            # For numeric columns, fill with median
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Convert object columns to category if they have few unique values
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            unique_count = df_clean[col].nunique()
            if unique_count <= 20:
                df_clean[col] = df_clean[col].astype('category')
    
    return df_clean


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive dataset summary.
    Returns dictionary with dataset statistics.
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist()
    }
    
    # Add basic statistics for numeric columns
    if summary['numeric_columns']:
        summary['numeric_stats'] = df[summary['numeric_columns']].describe().to_dict()
    
    # Add value counts for categorical columns
    if summary['categorical_columns']:
        summary['categorical_stats'] = {}
        for col in summary['categorical_columns']:
            summary['categorical_stats'][col] = df[col].value_counts().head(10).to_dict()
    
    return summary


def prepare_for_training(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, str]:
    """
    Prepare dataset for training by cleaning and detecting problem type.
    Returns (cleaned_dataframe, problem_type).
    """
    # Clean the dataset
    df_clean = clean_dataset(df)
    
    # Detect problem type
    problem_type = detect_problem_type(df_clean, target_col)
    
    return df_clean, problem_type


def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Detect if the problem is classification or regression.
    """
    unique_values = df[target_col].nunique()
    total_values = len(df[target_col])
    
    # If target has few unique values relative to dataset size, likely classification
    if unique_values <= 20 or unique_values / total_values < 0.1:
        return 'classification'
    else:
        return 'regression'


def validate_target_column(df: pd.DataFrame, target_col: str) -> Tuple[bool, str]:
    """
    Validate if the selected target column is suitable for training.
    Returns (is_valid, error_message).
    """
    if target_col not in df.columns:
        return False, f"Column '{target_col}' not found in dataset."
    
    # Check for too many missing values in target
    missing_percent = df[target_col].isnull().sum() / len(df) * 100
    if missing_percent > 30:
        return False, f"Target column has {missing_percent:.1f}% missing values. Please choose a different column."
    
    # Check for constant values
    if df[target_col].nunique() <= 1:
        return False, "Target column has only one unique value. Please choose a different column."
    
    return True, ""
