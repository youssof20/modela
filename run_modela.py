#!/usr/bin/env python3
"""
Modela - Simple Run Script
"""

import subprocess
import sys
import os

def main():
    """Run Modela application."""
    print("🤖 Starting Modela AutoML Platform...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found. Please run this script from the Modela directory.")
        return
    
    # Check if sample data exists
    if not os.path.exists("sample_data"):
        print("📊 Creating sample datasets...")
        try:
            subprocess.run([sys.executable, "create_demo_simple.py"], check=True)
            print("✅ Sample datasets created!")
        except subprocess.CalledProcessError:
            print("⚠️ Could not create sample datasets. You can still upload your own data.")
    
    print("🚀 Starting Streamlit application...")
    print("📱 Open your browser to: http://localhost:8501")
    print("=" * 50)
    
    try:
        # Run Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Modela stopped. Thanks for using Modela!")
    except Exception as e:
        print(f"❌ Error starting Modela: {e}")

if __name__ == "__main__":
    main()
