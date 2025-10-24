#!/usr/bin/env python3
"""
Simple demo data creator for Modela
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, load_iris
from pathlib import Path

def create_sample_datasets():
    """Create sample datasets for demonstration."""
    print("Creating sample datasets...")
    
    # Create data directory
    data_dir = Path("sample_data")
    data_dir.mkdir(exist_ok=True)
    
    # 1. Iris Dataset (Classification)
    print("Creating Iris dataset...")
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    iris_df.to_csv(data_dir / "iris.csv", index=False)
    print(f"Iris dataset created: {iris_df.shape}")
    
    # 2. Synthetic Classification Dataset
    print("Creating synthetic classification dataset...")
    X_class, y_class = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_classes=3, 
        n_clusters_per_class=1,
        random_state=42
    )
    class_df = pd.DataFrame(X_class, columns=[f'feature_{i}' for i in range(10)])
    class_df['target'] = y_class
    class_df.to_csv(data_dir / "synthetic_classification.csv", index=False)
    print(f"Synthetic classification dataset created: {class_df.shape}")
    
    # 3. Synthetic Regression Dataset
    print("Creating synthetic regression dataset...")
    X_reg, y_reg = make_regression(
        n_samples=1000, 
        n_features=10, 
        noise=0.1,
        random_state=42
    )
    reg_df = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(10)])
    reg_df['target'] = y_reg
    reg_df.to_csv(data_dir / "synthetic_regression.csv", index=False)
    print(f"Synthetic regression dataset created: {reg_df.shape}")
    
    print("\nDemo datasets created successfully!")
    print("Sample datasets available in 'sample_data/' directory:")
    for file in data_dir.glob("*.csv"):
        print(f"  - {file.name}")
    
    print("\nTo test Modela:")
    print("1. Run: streamlit run app.py")
    print("2. Go to Upload Dataset page")
    print("3. Upload any CSV file from sample_data/")
    print("4. Follow the training workflow")

if __name__ == "__main__":
    create_sample_datasets()
