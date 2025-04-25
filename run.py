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
    """Download the default YOLOv8m model"""
    print("Downloading the default YOLOv8m model...")
    try:
        subprocess.check_call([sys.executable, "download_models.py", "--models", "m", "--specialized"])
        print("Default models downloaded successfully!")
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
    if not os.path.exists("yolov8m.pt"):
        download_default_model()
    else:
        print("Default YOLOv8m model already exists.")
        
        # Ask about specialized models
        download_specialized = input("Do you want to download specialized models for better indoor detection? (y/n): ").lower()
        if download_specialized.startswith('y'):
            try:
                subprocess.check_call([sys.executable, "download_models.py", "--specialized"])
                print("Specialized models downloaded successfully!")
            except subprocess.CalledProcessError as e:
                print(f"Error downloading specialized models: {e}")
    
    # Run the application
    run_app() 