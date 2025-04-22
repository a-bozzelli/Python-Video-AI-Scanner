import os
import argparse
from ultralytics import YOLO

def download_models(selected_models=None):
    """
    Download YOLOv8 models
    
    Args:
        selected_models (list, optional): List of models to download.
            If None, downloads the default (nano) model only.
    """
    # Define available models
    available_models = {
        "n": "yolov8n.pt",  # nano
        "s": "yolov8s.pt",  # small
        "m": "yolov8m.pt",  # medium
        "l": "yolov8l.pt",  # large
        "x": "yolov8x.pt",  # extra large
    }
    
    # If no specific models selected, default to nano
    if not selected_models:
        selected_models = ["n"]
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download each model
    for model_key in selected_models:
        if model_key in available_models:
            model_name = available_models[model_key]
            print(f"Downloading {model_name}...")
            
            # YOLO() will automatically download the model if not found
            model = YOLO(model_name)
            print(f"Successfully downloaded {model_name}")
        else:
            print(f"Unknown model: {model_key}. Skipping.")

if __name__ == "__main__":
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Download YOLOv8 models")
    parser.add_argument(
        "--models", 
        nargs="+", 
        choices=["n", "s", "m", "l", "x", "all"],
        default=["n"],
        help="Models to download (n=nano, s=small, m=medium, l=large, x=extra large, all=all models)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if "all" is specified
    if "all" in args.models:
        models_to_download = ["n", "s", "m", "l", "x"]
    else:
        models_to_download = args.models
    
    # Download the models
    download_models(models_to_download) 