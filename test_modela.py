#!/usr/bin/env python3
"""
Modela Test Script
Simple test script to verify Modela installation and functionality.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pycaret
        print("✅ PyCaret imported successfully")
    except ImportError as e:
        print(f"❌ PyCaret import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import plotly
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    return True

def test_data_creation():
    """Test creating sample data."""
    print("📊 Testing data creation...")
    
    try:
        # Create sample classification data
        np.random.seed(42)
        n_samples = 100
        n_features = 4
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['target'] = y
        
        print(f"✅ Created sample dataset: {df.shape}")
        print(f"   - Features: {n_features}")
        print(f"   - Samples: {n_samples}")
        print(f"   - Target distribution: {df['target'].value_counts().to_dict()}")
        
        return True
    except Exception as e:
        print(f"❌ Data creation failed: {e}")
        return False

def test_file_structure():
    """Test if required files exist."""
    print("📁 Testing file structure...")
    
    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        "LICENSE",
        "pages/1_upload.py",
        "pages/2_train.py",
        "pages/3_results.py",
        "pages/4_projects.py",
        "utils/automl.py",
        "utils/firebase_client.py",
        "utils/preprocessing.py",
        "utils/visualization.py",
        ".streamlit/config.toml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    return True

def test_directories():
    """Test if required directories exist."""
    print("📂 Testing directories...")
    
    required_dirs = [
        "pages",
        "utils",
        ".streamlit",
        "data"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ Missing directory: {dir_path}/")
            return False
    
    return True

def test_pycaret_functionality():
    """Test basic PyCaret functionality."""
    print("🤖 Testing PyCaret functionality...")
    
    try:
        from pycaret.classification import setup, compare_models
        from sklearn.datasets import make_classification
        
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
        df['target'] = y
        
        # Test PyCaret setup
        clf = setup(data=df, target='target', silent=True, verbose=False)
        print("✅ PyCaret setup completed")
        
        # Test model comparison (limited for speed)
        best_model = compare_models(include=['lr', 'rf'], n_select=1, verbose=False)
        print("✅ PyCaret model comparison completed")
        
        return True
    except Exception as e:
        print(f"❌ PyCaret test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Modela AutoML Platform Test Suite")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Directories", test_directories),
        ("Imports", test_imports),
        ("Data Creation", test_data_creation),
        ("PyCaret Functionality", test_pycaret_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Modela is ready to use.")
        print("\n🚀 To start the application:")
        print("   streamlit run app.py")
        print("\n🌐 Then open: http://localhost:8501")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
