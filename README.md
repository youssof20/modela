# Modela - Open Source AutoML Platform

[![GitHub stars](https://img.shields.io/github/stars/youssof20/modela?style=social)](https://github.com/youssof20/modela)
[![GitHub forks](https://img.shields.io/github/forks/youssof20/modela?style=social)](https://github.com/youssof20/modela)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Modela is a powerful, open-source AutoML platform that democratizes machine learning by making it accessible to everyone. Built with Streamlit and PyCaret, it provides an intuitive web interface for training machine learning models without writing a single line of code.

## Features

- **📊 Easy Data Upload**: Support for CSV and Excel files up to 50MB
- **🤖 AutoML Training**: Automatic model training using PyCaret with 5+ algorithms
- **📈 Rich Visualizations**: Feature importance, confusion matrices, and performance metrics
- **💾 Model Download**: Download trained models as .pkl files

## 🚀 Quick Start

**Modela is designed to run locally on your machine.** This gives you full control over your data and models while keeping everything private and secure.

### Local Installation (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/youssof20/modela.git
   cd modela
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python run_modela.py
   ```
   
   Or manually:
   ```bash
   python -m streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

### System Requirements

- **Python**: 3.8, 3.9, or 3.10 (3.11+ may have compatibility issues)
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: 2GB free space
- **OS**: Windows, macOS, or Linux

### Alternative Deployment Options

**Docker (For Production/Sharing)**
```bash
# Build and run with Docker
docker-compose up
# Access at http://localhost:8501
```

**Cloud Deployment (Not Recommended)**
- **Note**: PyCaret has compatibility issues with newer Python versions
- **Alternative**: Use Docker for consistent deployment
- **Local First**: Designed to run on your machine for privacy and performance

## 📊 Supported Data Types

- **Classification**: Predicting categories (e.g., spam/not spam, customer churn)
- **Regression**: Predicting numerical values (e.g., house prices, sales forecasts)

## 🔧 Configuration

### Local Storage

The application automatically creates a `data/` directory to store:
- User accounts and authentication data
- Uploaded datasets
- Trained models and metadata

No additional configuration is required!

## Usage Guide

### 1. Upload Dataset
- Click "Upload Dataset" in the sidebar
- Choose a CSV or Excel file (max 50MB)
- Preview your data and check the summary
- Save the dataset to local storage

### 2. Train Model
- Select your target column (what you want to predict)
- Choose problem type (classification or regression)
- Configure advanced settings (optional)
- Click "Start Training" and wait 1-3 minutes

### 3. View Results
- See model performance metrics
- View feature importance charts
- Download your trained model
- Get insights and explanations

### 4. Manage Projects
- View all your trained models
- Download previous models
- Track your progress over time
---
