"""
Visualization utilities for model results and insights.
Creates charts and plots for the results dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import streamlit as st


def plot_feature_importance(feature_importance: Dict[str, float], title: str = "Feature Importance") -> go.Figure:
    """
    Create a horizontal bar chart for feature importance.
    """
    if not feature_importance:
        # Create empty figure if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No feature importance data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, scores = zip(*sorted_features[:10])  # Top 10 features
    
    fig = go.Figure(data=[
        go.Bar(
            y=list(features),
            x=list(scores),
            orientation='h',
            marker_color='#1f77b4',
            text=[f"{score:.3f}" for score in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400,
        margin=dict(l=100, r=50, t=50, b=50),
        showlegend=False
    )
    
    return fig


def plot_confusion_matrix(y_true: List, y_pred: List, class_names: List[str] = None) -> go.Figure:
    """
    Create a confusion matrix heatmap.
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def plot_regression_scatter(y_true: List, y_pred: List) -> go.Figure:
    """
    Create a scatter plot for regression predictions vs actual values.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(
            color='#1f77b4',
            size=8,
            opacity=0.6
        ),
        name='Predictions'
    ))
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title="Predictions vs Actual Values",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def plot_metrics_comparison(metrics: Dict[str, float], problem_type: str) -> go.Figure:
    """
    Create a bar chart comparing different metrics.
    """
    if problem_type == 'classification':
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1', 0),
            metrics.get('auc', 0)
        ]
    else:
        metric_names = ['RMSE', 'MAE', 'R²', 'MAPE']
        metric_values = [
            metrics.get('rmse', 0),
            metrics.get('mae', 0),
            metrics.get('r2', 0),
            metrics.get('mape', 0)
        ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color='#2E8B57',
            text=[f"{val:.3f}" for val in metric_values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Metrics",
        yaxis_title="Score",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False
    )
    
    return fig


def plot_training_progress(progress_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a progress chart showing training stages.
    """
    stages = [item['stage'] for item in progress_data]
    progress = [item['progress'] for item in progress_data]
    
    fig = go.Figure(data=[
        go.Bar(
            x=stages,
            y=progress,
            marker_color='#FF6B6B',
            text=[f"{p}%" for p in progress],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Training Progress",
        xaxis_title="Training Stage",
        yaxis_title="Progress (%)",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False
    )
    
    return fig


def plot_dataset_overview(df: pd.DataFrame) -> go.Figure:
    """
    Create an overview chart of the dataset.
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Dataset Shape', 'Data Types', 'Missing Values', 'Memory Usage'),
        specs=[[{"type": "indicator"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # Dataset shape
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=df.shape[0],
            title={"text": "Rows"},
            domain={'x': [0, 0.5], 'y': [0.5, 1]}
        ),
        row=1, col=1
    )
    
    # Data types
    dtype_counts = df.dtypes.value_counts()
    fig.add_trace(
        go.Pie(
            labels=dtype_counts.index.astype(str),
            values=dtype_counts.values,
            name="Data Types"
        ),
        row=1, col=2
    )
    
    # Missing values
    missing_counts = df.isnull().sum().sort_values(ascending=False).head(10)
    fig.add_trace(
        go.Bar(
            x=missing_counts.index,
            y=missing_counts.values,
            name="Missing Values"
        ),
        row=2, col=1
    )
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=memory_mb,
            title={"text": "Memory Usage (MB)"},
            domain={'x': [0, 0.5], 'y': [0, 0.5]}
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Dataset Overview",
        height=600,
        showlegend=False
    )
    
    return fig


def create_metrics_summary_card(metrics: Dict[str, float], problem_type: str) -> str:
    """
    Create a formatted summary card for metrics display.
    """
    if problem_type == 'classification':
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0)
        
        summary = f"""
        **🎯 Model Performance Summary**
        
        - **Accuracy**: {accuracy:.1%} - Your model correctly predicts {accuracy:.1%} of all cases
        - **Precision**: {precision:.1%} - When your model predicts positive, it's right {precision:.1%} of the time
        - **Recall**: {recall:.1%} - Your model catches {recall:.1%} of all positive cases
        - **F1-Score**: {f1:.1%} - Balanced measure of precision and recall
        """
    else:
        rmse = metrics.get('rmse', 0)
        mae = metrics.get('mae', 0)
        r2 = metrics.get('r2', 0)
        
        summary = f"""
        **🎯 Model Performance Summary**
        
        - **RMSE**: {rmse:.3f} - Root mean square error (lower is better)
        - **MAE**: {mae:.3f} - Mean absolute error (lower is better)
        - **R²**: {r2:.1%} - Coefficient of determination (higher is better)
        """
    
    return summary
