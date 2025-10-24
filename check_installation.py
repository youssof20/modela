#!/usr/bin/env python3
"""
Modela Installation Checker
Checks Python version compatibility and provides installation guidance.
"""

import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible with PyCaret."""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor in [8, 9, 10]:
        print("✅ Python version is compatible with PyCaret!")
        return True
    elif version.major == 3 and version.minor >= 11:
        print("❌ Python 3.11+ has compatibility issues with PyCaret")
        print("💡 Recommended solutions:")
        print("   1. Use Python 3.10: https://www.python.org/downloads/")
        print("   2. Use Docker: docker-compose up")
        print("   3. Use conda: conda create -n modela python=3.10")
        return False
    else:
        print("❌ Python 3.8+ is required")
        return False

def check_pip():
    """Check if pip is available."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False

def suggest_installation():
    """Suggest installation steps based on the system."""
    print("\n🚀 Installation Steps:")
    print("=" * 50)
    
    if platform.system() == "Windows":
        print("1. Create virtual environment:")
        print("   python -m venv modela_env")
        print("   modela_env\\Scripts\\activate")
        print("\n2. Install dependencies:")
        print("   pip install -r requirements.txt")
        print("\n3. Run the application:")
        print("   streamlit run app.py")
    else:
        print("1. Create virtual environment:")
        print("   python -m venv modela_env")
        print("   source modela_env/bin/activate")
        print("\n2. Install dependencies:")
        print("   pip install -r requirements.txt")
        print("\n3. Run the application:")
        print("   streamlit run app.py")

def main():
    """Main installation checker."""
    print("🔍 Modela Installation Checker")
    print("=" * 50)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check pip
    pip_ok = check_pip()
    
    print("\n" + "=" * 50)
    
    if python_ok and pip_ok:
        print("🎉 Your system is ready for Modela!")
        suggest_installation()
    else:
        print("⚠️  Please fix the issues above before installing Modela")
        print("\n💡 Alternative: Use Docker for guaranteed compatibility")
        print("   docker-compose up")

if __name__ == "__main__":
    main()
