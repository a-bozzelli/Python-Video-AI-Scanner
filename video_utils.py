import cv2
import numpy as np
import os
import tempfile
from collections import Counter
from ultralytics import YOLO

def extract_frames(video_path, output_dir=None, frame_interval=24):
    """
    Extract frames from a video at specified intervals
    
    Args:
        video_path (str): Path to the video file
        output_dir (str, optional): Directory to save extracted frames
        frame_interval (int): Extract every nth frame
        
    Returns:
        list: Paths to extracted frames if output_dir is provided, 
              or list of numpy arrays if output_dir is None
    """
    # Create output directory if specified and doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Adjust frame_interval based on fps if needed
    if fps > 30:
        frame_interval = int(frame_interval * (fps / 30))
    
    # Initialize list to store frames or paths
    frames = []
    
    # Read and extract frames
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every nth frame
        if frame_idx % frame_interval == 0:
            if output_dir is not None:
                # Save the frame to the output directory
                frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
            else:
                # Convert BGR to RGB if not saving to disk
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        frame_idx += 1
    
    # Release the video capture
    cap.release()
    
    return frames

def detect_objects_in_frames(model, frames, conf_threshold=0.25):
    """
    Detect objects in a list of frames
    
    Args:
        model: YOLO model
        frames (list): List of frames (paths or numpy arrays)
        conf_threshold (float): Confidence threshold for detections
        
    Returns:
        dict: Dictionary with detected objects and their counts
    """
    # Dictionary to store detected objects and counts
    detected_objects = {}
    
    # Process each frame
    for frame in frames:
        # Run inference
        if isinstance(frame, str):
            # If frame is a path, let YOLO handle loading
            results = model(frame)
        else:
            # If frame is a numpy array, use it directly
            results = model(frame)
        
        # Process results of current frame
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get the class name and confidence
                cls_id = int(box.cls.item())
                class_name = model.names[cls_id]
                conf = box.conf.item()
                
                # Only count detections above threshold
                if conf > conf_threshold:
                    if class_name in detected_objects:
                        detected_objects[class_name] += 1
                    else:
                        detected_objects[class_name] = 1
    
    return detected_objects

def analyze_video(video_path, model, frame_interval=24, conf_threshold=0.25, save_annotated=False, output_dir=None):
    """
    Analyze a video to detect objects
    
    Args:
        video_path (str): Path to the video file
        model: YOLO model
        frame_interval (int): Analyze every nth frame
        conf_threshold (float): Confidence threshold for detections
        save_annotated (bool): Whether to save annotated frames
        output_dir (str, optional): Directory to save annotated frames
        
    Returns:
        tuple: (detected_objects, annotated_frames_dir)
    """
    # Create temporary directory for frames if needed
    if save_annotated and output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    # Extract frames
    frames = extract_frames(video_path, None, frame_interval)
    
    # Dictionary to store detected objects and counts
    detected_objects = {}
    
    # Process each frame
    for i, frame in enumerate(frames):
        # Run inference
        results = model(frame)
        
        # Process results of current frame
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get the class name and confidence
                cls_id = int(box.cls.item())
                class_name = model.names[cls_id]
                conf = box.conf.item()
                
                # Only count detections above threshold
                if conf > conf_threshold:
                    if class_name in detected_objects:
                        detected_objects[class_name] += 1
                    else:
                        detected_objects[class_name] = 1
        
        # Save annotated frame if requested
        if save_annotated:
            annotated_frame = results[0].plot()
            frame_path = os.path.join(output_dir, f"annotated_{i:06d}.jpg")
            cv2.imwrite(frame_path, annotated_frame)
    
    # Return results
    if save_annotated:
        return detected_objects, output_dir
    else:
        return detected_objects, None

def create_summary_video(video_path, model, output_path, frame_interval=24, conf_threshold=0.25):
    """
    Create a summary video with object detections
    
    Args:
        video_path (str): Path to the video file
        model: YOLO model
        output_path (str): Path to save the summary video
        frame_interval (int): Process every nth frame
        conf_threshold (float): Confidence threshold for detections
        
    Returns:
        str: Path to the summary video
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps/frame_interval, (width, height))
    
    # Process frames
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every nth frame
        if frame_idx % frame_interval == 0:
            # Run inference
            results = model(frame)
            
            # Plot results on frame
            annotated_frame = results[0].plot()
            
            # Write to output video
            out.write(annotated_frame)
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    
    return output_path

if __name__ == "__main__":
    # Example usage
    model = YOLO("yolov8n.pt")
    video_path = "sample_video.mp4"
    
    # Analyze video
    objects, _ = analyze_video(video_path, model)
    print("Detected objects:", objects)
    
    # Create summary video
    summary_path = create_summary_video(video_path, model, "summary.mp4")
    print(f"Summary video saved to: {summary_path}") 