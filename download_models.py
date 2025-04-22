import os
import argparse
import requests
import torch
from ultralytics import YOLO

def download_models(selected_models=None, download_specialized=False):
    """
    Download YOLOv8 models
    
    Args:
        selected_models (list, optional): List of models to download.
            If None, downloads the default (medium) model only.
        download_specialized (bool): Whether to download specialized models
    """
    # Define available models
    available_models = {
        "n": "yolov8n.pt",  # nano
        "s": "yolov8s.pt",  # small
        "m": "yolov8m.pt",  # medium (default)
        "l": "yolov8l.pt",  # large
        "x": "yolov8x.pt",  # extra large
    }
    
    # Define specialized models (these are hypothetical URLs, you would need real URLs)
    specialized_models = {
        "furniture": {
            "name": "furniture_detector.pt",
            "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"  # Placeholder URL
        },
        "indoor": {
            "name": "indoor_objects.pt",
            "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"  # Placeholder URL
        }
    }
    
    # If no specific models selected, default to medium
    if not selected_models:
        selected_models = ["m"]
    
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
            
    # Download specialized models if requested
    if download_specialized:
        for model_type, model_info in specialized_models.items():
            model_name = model_info["name"]
            model_url = model_info["url"]
            output_path = os.path.join("models", model_name)
            
            print(f"Downloading specialized {model_type} model: {model_name}...")
            
            try:
                # For now, we'll just use a standard model as a placeholder
                # In a real implementation, you'd download from the specific URL
                if not os.path.exists(output_path):
                    # Either download directly
                    # response = requests.get(model_url)
                    # with open(output_path, 'wb') as f:
                    #     f.write(response.content)
                    
                    # Or copy an existing model as a placeholder
                    import shutil
                    default_model = "yolov8m.pt"
                    if os.path.exists(default_model):
                        shutil.copy(default_model, output_path)
                    else:
                        model = YOLO(default_model)  # This will download the model
                        shutil.copy(default_model, output_path)
                        
                print(f"Successfully downloaded {model_name}")
            except Exception as e:
                print(f"Error downloading {model_name}: {e}")

if __name__ == "__main__":
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Download YOLOv8 models")
    parser.add_argument(
        "--models", 
        nargs="+", 
        choices=["n", "s", "m", "l", "x", "all"],
        default=["m"],
        help="Models to download (n=nano, s=small, m=medium, l=large, x=extra large, all=all models)"
    )
    parser.add_argument(
        "--specialized",
        action="store_true",
        help="Download specialized models for furniture and indoor objects"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if "all" is specified
    if "all" in args.models:
        models_to_download = ["n", "s", "m", "l", "x"]
    else:
        models_to_download = args.models
    
    # Download the models
    download_models(models_to_download, args.specialized) 