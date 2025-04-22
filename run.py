import os
import sys
import subprocess
import platform

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import streamlit
        import ultralytics
        import cv2
        import numpy
        import coremltools
        from PIL import Image
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        return False

def install_requirements():
    """Install the required packages from requirements.txt"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Requirements installed successfully!")

def download_default_model():
    """Download the default YOLOv8n model"""
    print("Downloading the default YOLOv8n model...")
    try:
        subprocess.check_call([sys.executable, "download_models.py"])
        print("Default model downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

def run_app():
    """Run the Streamlit application"""
    print("Starting Home Video Scanner...")
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    print("=== Home Video Scanner Setup ===")
    
    # Check if requirements are already installed
    if not check_requirements():
        print("Some required packages are missing.")
        install_requirements()
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download the default model if not already present
    if not os.path.exists("yolov8n.pt"):
        download_default_model()
    
    # Run the application
    run_app() 