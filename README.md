# Modela - AutoML Web App

A simple web interface for training machine learning models without code. Built with Streamlit and PyCaret.

## What it does

Upload a CSV, pick what you want to predict, and it trains a bunch of ML models for you. Shows you which one works best and lets you download it.

## Setup

```bash
git clone https://github.com/youssof20/modela.git
cd modela
pip install -r requirements.txt
python run_modela.py
```

Go to `http://localhost:8501`

## Requirements

- Python 3.8-3.10 (PyCaret doesn't work great with 3.11+)
- 4GB RAM minimum
- CSV or Excel file with your data

## How to use

1. Upload your dataset (CSV or Excel)
2. Pick which column you want to predict
3. Choose classification or regression
4. Click "Start Training" and wait a few minutes
5. Download your trained model

## What you get

- Comparison of different ML algorithms
- Performance metrics and charts
- Feature importance visualization
- Trained model as a .pkl file

## Limitations

- Max 50MB file size
- Takes a while to train (1-3 minutes usually)
- Some Python version compatibility issues
- Works best locally, cloud deployment is tricky

## Tech

- Streamlit for the web interface
- PyCaret for the AutoML part
- Saves everything locally in a `data/` folder
