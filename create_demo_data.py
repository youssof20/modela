#!/usr/bin/env python3
"""
Modela Demo Script
Demonstration script with sample datasets for testing Modela functionality.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, load_iris, load_wine
from pathlib import Path
import os

def create_sample_datasets():
    """Create sample datasets for demonstration."""
    print("📊 Creating sample datasets...")
    
    # Create data directory
    data_dir = Path("sample_data")
    data_dir.mkdir(exist_ok=True)
    
    # 1. Iris Dataset (Classification)
    print("🌺 Creating Iris dataset...")
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    iris_df.to_csv(data_dir / "iris.csv", index=False)
    print(f"✅ Iris dataset created: {iris_df.shape}")
    
    # 2. Wine Dataset (Classification)
    print("🍷 Creating Wine dataset...")
    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df['wine_type'] = wine.target
    wine_df.to_csv(data_dir / "wine.csv", index=False)
    print(f"✅ Wine dataset created: {wine_df.shape}")
    
    # 3. Synthetic Classification Dataset
    print("🎯 Creating synthetic classification dataset...")
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
    print(f"✅ Synthetic classification dataset created: {class_df.shape}")
    
    # 4. Synthetic Regression Dataset
    print("📈 Creating synthetic regression dataset...")
    X_reg, y_reg = make_regression(
        n_samples=1000, 
        n_features=10, 
        noise=0.1,
        random_state=42
    )
    reg_df = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(10)])
    reg_df['target'] = y_reg
    reg_df.to_csv(data_dir / "synthetic_regression.csv", index=False)
    print(f"✅ Synthetic regression dataset created: {reg_df.shape}")
    
    # 5. Customer Churn Dataset (Business Example)
    print("👥 Creating customer churn dataset...")
    np.random.seed(42)
    n_customers = 1000
    
    churn_df = pd.DataFrame({
        'age': np.random.normal(35, 10, n_customers).astype(int),
        'tenure': np.random.exponential(2, n_customers).astype(int),
        'monthly_charges': np.random.normal(70, 20, n_customers),
        'total_charges': np.random.normal(2000, 500, n_customers),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_customers),
        'churn': np.random.choice([0, 1], n_customers, p=[0.7, 0.3])
    })
    
    # Make churn more realistic based on features
    churn_df.loc[churn_df['contract_type'] == 'Month-to-month', 'churn'] = np.random.choice([0, 1], 
        size=len(churn_df[churn_df['contract_type'] == 'Month-to-month']), p=[0.5, 0.5])
    
    churn_df.to_csv(data_dir / "customer_churn.csv", index=False)
    print(f"✅ Customer churn dataset created: {churn_df.shape}")
    
    # 6. House Prices Dataset (Regression Example)
    print("🏠 Creating house prices dataset...")
    np.random.seed(42)
    n_houses = 1000
    
    house_df = pd.DataFrame({
        'size_sqft': np.random.normal(2000, 500, n_houses).astype(int),
        'bedrooms': np.random.poisson(3, n_houses),
        'bathrooms': np.random.poisson(2, n_houses),
        'age_years': np.random.exponential(10, n_houses).astype(int),
        'distance_to_city': np.random.exponential(5, n_houses),
        'school_rating': np.random.uniform(1, 10, n_houses),
        'crime_rate': np.random.uniform(0, 100, n_houses),
        'price': 0  # Will be calculated
    })
    
    # Calculate realistic prices
    house_df['price'] = (
        house_df['size_sqft'] * 100 +
        house_df['bedrooms'] * 10000 +
        house_df['bathrooms'] * 15000 -
        house_df['age_years'] * 1000 -
        house_df['distance_to_city'] * 5000 +
        house_df['school_rating'] * 5000 -
        house_df['crime_rate'] * 100 +
        np.random.normal(0, 20000, n_houses)
    ).astype(int)
    
    house_df.to_csv(data_dir / "house_prices.csv", index=False)
    print(f"✅ House prices dataset created: {house_df.shape}")
    
    return data_dir

def create_dataset_descriptions():
    """Create descriptions for each dataset."""
    print("📝 Creating dataset descriptions...")
    
    descriptions = {
        "iris.csv": {
            "name": "Iris Flower Classification",
            "description": "Classify iris flowers into three species based on petal and sepal measurements.",
            "target": "species",
            "problem_type": "classification",
            "difficulty": "Easy"
        },
        "wine.csv": {
            "name": "Wine Quality Classification",
            "description": "Classify wines into different categories based on chemical properties.",
            "target": "wine_type",
            "problem_type": "classification",
            "difficulty": "Easy"
        },
        "synthetic_classification.csv": {
            "name": "Synthetic Classification",
            "description": "A synthetic dataset with 3 classes and 10 features for testing classification algorithms.",
            "target": "target",
            "problem_type": "classification",
            "difficulty": "Medium"
        },
        "synthetic_regression.csv": {
            "name": "Synthetic Regression",
            "description": "A synthetic dataset with continuous target variable for testing regression algorithms.",
            "target": "target",
            "problem_type": "regression",
            "difficulty": "Medium"
        },
        "customer_churn.csv": {
            "name": "Customer Churn Prediction",
            "description": "Predict whether customers will churn based on their demographics and usage patterns.",
            "target": "churn",
            "problem_type": "classification",
            "difficulty": "Medium"
        },
        "house_prices.csv": {
            "name": "House Price Prediction",
            "description": "Predict house prices based on size, location, and other features.",
            "target": "price",
            "problem_type": "regression",
            "difficulty": "Medium"
        }
    }
    
    # Save descriptions to JSON
    import json
    with open("sample_data/dataset_descriptions.json", "w") as f:
        json.dump(descriptions, f, indent=2)
    
    print("✅ Dataset descriptions created")

def main():
    """Main demo function."""
    print("🎯 Modela AutoML Platform - Demo Data Creator")
    print("=" * 60)
    
    # Create sample datasets
    data_dir = create_sample_datasets()
    
    # Create descriptions
    create_dataset_descriptions()
    
    print("\n🎉 Demo datasets created successfully!")
    print("=" * 60)
    print("📁 Sample datasets available in 'sample_data/' directory:")
    
    for file in data_dir.glob("*.csv"):
        print(f"   📊 {file.name}")
    
    print("\n🚀 To test Modela with these datasets:")
    print("1. Run: streamlit run app.py")
    print("2. Go to Upload Dataset page")
    print("3. Upload any CSV file from sample_data/")
    print("4. Follow the training workflow")
    
    print("\n💡 Recommended testing order:")
    print("   1. iris.csv (easy classification)")
    print("   2. house_prices.csv (regression)")
    print("   3. customer_churn.csv (business case)")
    
    print("\n📚 Dataset descriptions saved to: sample_data/dataset_descriptions.json")

if __name__ == "__main__":
    main()
