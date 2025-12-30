#!/usr/bin/env python3
"""
AutoIntel Setup Script
Automated installation and setup for the AutoIntel platform
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.10+")
        return False

def main():
    """Main setup function"""
    print("🚗 AutoIntel - Car Market Intelligence Platform Setup")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found. Please run this script from the AutoIntel project root.")
        sys.exit(1)

    # Create virtual environment
    if not Path("venv").exists():
        if not run_command("python -m venv venv", "Creating virtual environment"):
            sys.exit(1)
    else:
        print("ℹ️  Virtual environment already exists")

    # Activate virtual environment and install dependencies
    activate_cmd = "venv\\Scripts\\activate" if os.name == 'nt' else "source venv/bin/activate"

    # Install requirements
    if not run_command(f"{activate_cmd} && pip install -r requirements.txt", "Installing Python dependencies"):
        sys.exit(1)

    # Install Playwright browsers
    if not run_command(f"{activate_cmd} && playwright install", "Installing Playwright browsers"):
        sys.exit(1)

    # Verify installation
    print("\n🔍 Verifying installation...")
    try:
        # Test imports
        test_imports = [
            ("streamlit", "Streamlit"),
            ("pandas", "Pandas"),
            ("sklearn", "Scikit-learn"),
            ("playwright", "Playwright"),
            ("plotly", "Plotly")
        ]

        for module, name in test_imports:
            try:
                __import__(module)
                print(f"✅ {name} - OK")
            except ImportError:
                print(f"❌ {name} - FAILED")
                return False

        print("\n🎉 AutoIntel setup completed successfully!")
        print("\n🚀 To start the dashboard, run:")
        print("   streamlit run dashboard/app.py")
        print("\n📱 Then open: http://localhost:8501")

        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 Setup failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n✨ Happy analyzing! 🚗💨")
