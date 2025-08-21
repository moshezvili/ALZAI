#!/usr/bin/env python
"""
Setup and installation script for the Clinical ML Pipeline.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is adequate."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def setup_environment():
    """Set up the development environment."""
    print("🚀 Setting up Clinical ML Pipeline Environment\n")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "mlruns",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Set environment variables
    env_vars = {
        "PYTHONPATH": str(Path.cwd()),
        "MLFLOW_TRACKING_URI": "file:./mlruns"
    }
    
    print("\n🔧 Environment variables to set:")
    for key, value in env_vars.items():
        print(f"export {key}={value}")
    
    # Generate sample data
    if run_command(
        "python src/data_generation/generate_clinical_data.py --num_patients 1000 --years_per_patient 3",
        "Generating sample clinical data"
    ):
        print("📊 Sample data generated successfully")
    
    return True

def run_tests():
    """Run the test suite."""
    print("\n🧪 Running tests...")
    return run_command("python -m pytest tests/ -v", "Running test suite")

def main():
    """Main setup function."""
    if not setup_environment():
        print("\n❌ Setup failed!")
        return 1
    
    print("\n✅ Environment setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Set the environment variables shown above")
    print("2. Run: python src/pipeline/training_pipeline.py --config config/training_config.yaml --data data/raw/clinical_data.parquet --output models")
    print("3. Start serving: python src/serving/api.py")
    print("4. View API docs at: http://localhost:8000/docs")
    print("5. Run MLflow UI: mlflow ui")
    
    # Ask if user wants to run tests
    response = input("\n🧪 Would you like to run tests now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        if run_tests():
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed")
    
    return 0

if __name__ == "__main__":
    exit(main())
