#!/usr/bin/env python3
"""
Modela Setup Script
Automated setup and installation script for Modela AutoML platform.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please install Python 3.8 or higher")
        return False

def install_dependencies():
    """Install Python dependencies."""
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Install additional development dependencies
    dev_deps = ["pytest", "flake8", "mypy", "black"]
    for dep in dev_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")
    
    return True

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = ["data", "data/users", ".streamlit"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def setup_streamlit_config():
    """Setup Streamlit configuration."""
    print("⚙️ Setting up Streamlit configuration...")
    
    config_content = """[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 50
"""
    
    config_path = Path(".streamlit/config.toml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print("✅ Streamlit configuration created")
    return True

def test_installation():
    """Test the installation."""
    print("🧪 Testing installation...")
    
    # Test imports
    try:
        import streamlit
        import pycaret
        import pandas
        import numpy
        import plotly
        print("✅ All dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test Streamlit
    if run_command("streamlit --version", "Testing Streamlit"):
        return True
    else:
        return False

def main():
    """Main setup function."""
    print("🚀 Welcome to Modela AutoML Platform Setup!")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Setup Streamlit config
    if not setup_streamlit_config():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("=" * 50)
    print("📋 Next steps:")
    print("1. Run the application: streamlit run app.py")
    print("2. Open your browser to: http://localhost:8501")
    print("3. Start training your first model!")
    print("\n📚 Documentation: https://github.com/youssof20/modela")
    print("🐛 Issues: https://github.com/youssof20/modela/issues")
    print("💬 Discussions: https://github.com/youssof20/modela/discussions")

if __name__ == "__main__":
    main()
