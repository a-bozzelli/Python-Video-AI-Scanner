import os
from ultralytics import YOLO
import coremltools as ct

def convert_yolo_to_coreml(model_path="yolov8n.pt", output_path="models", model_name=None):
    """
    Convert a YOLOv8 model to CoreML format
    
    Args:
        model_path (str): Path to the YOLOv8 model (.pt file)
        output_path (str): Directory to save the CoreML model
        model_name (str, optional): Name for the output model file. If None, derived from model_path
        
    Returns:
        str: Path to the converted CoreML model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Load the model
    model = YOLO(model_path)
    
    # If model_name not provided, derive it from model_path
    if model_name is None:
        model_name = os.path.splitext(os.path.basename(model_path))[0] + ".mlmodel"
    elif not model_name.endswith(".mlmodel"):
        model_name += ".mlmodel"
    
    # Path for the output CoreML model
    output_model_path = os.path.join(output_path, model_name)
    
    # Export to CoreML
    model.export(format="coreml", imgsz=640)
    
    # The export method saves the model to a default location
    # We need to find it and potentially move it
    default_export_path = model_path.replace(".pt", "_saved_model")
    coreml_path = model_path.replace(".pt", ".mlmodel")
    
    # If the model was saved to a different location than requested, move it
    if os.path.exists(coreml_path) and output_model_path != coreml_path:
        import shutil
        shutil.move(coreml_path, output_model_path)
        
    return output_model_path

def optimize_for_device(model_path, output_path=None, device="neural_engine"):
    """
    Optimize a CoreML model for a specific device
    
    Args:
        model_path (str): Path to the CoreML model
        output_path (str, optional): Path to save the optimized model
        device (str): Device target ("neural_engine", "cpu", etc.)
        
    Returns:
        str: Path to the optimized model
    """
    # Load the model
    model = ct.models.MLModel(model_path)
    
    # Set the default output path if not provided
    if output_path is None:
        name, ext = os.path.splitext(model_path)
        output_path = f"{name}_{device}{ext}"
    
    # Optimize for the target device
    optimized_model = ct.optimize.neural_network(model, device)
    
    # Save the optimized model
    optimized_model.save(output_path)
    
    return output_path

if __name__ == "__main__":
    # Example usage
    coreml_model_path = convert_yolo_to_coreml()
    print(f"CoreML model saved to: {coreml_model_path}")
    
    # Optimize for Neural Engine (Apple Silicon)
    optimized_model_path = optimize_for_device(coreml_model_path)
    print(f"Optimized model saved to: {optimized_model_path}") 