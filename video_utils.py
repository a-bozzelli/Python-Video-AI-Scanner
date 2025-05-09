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

def analyze_video(video_path, model, frame_interval=24, conf_threshold=0.25, iou_threshold=0.45, 
               save_annotated=False, output_dir=None, img_size=640, class_filter=None):
    """
    Analyze a video to detect objects
    
    Args:
        video_path (str): Path to the video file
        model: YOLO model
        frame_interval (int): Analyze every nth frame
        conf_threshold (float): Confidence threshold for detections
        iou_threshold (float): IoU threshold for Non-Maximum Suppression
        save_annotated (bool): Whether to save annotated frames
        output_dir (str, optional): Directory to save annotated frames
        img_size (int): Image size for inference
        class_filter (list, optional): List of class names to include
        
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
        # Run inference with advanced parameters
        results = model(frame, conf=conf_threshold, iou=iou_threshold, imgsz=img_size)
        
        # Process results of current frame
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get the class name and confidence
                cls_id = int(box.cls.item())
                class_name = model.names[cls_id]
                conf = box.conf.item()
                
                # Apply class filter if provided
                if class_filter and class_name.lower() not in class_filter:
                    continue
                
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

def create_summary_video(video_path, model, output_path, frame_interval=24, conf_threshold=0.25,
                        iou_threshold=0.45, img_size=640, class_filter=None):
    """
    Create a summary video with object detections
    
    Args:
        video_path (str): Path to the video file
        model: YOLO model
        output_path (str): Path to save the summary video
        frame_interval (int): Process every nth frame
        conf_threshold (float): Confidence threshold for detections
        iou_threshold (float): IoU threshold for Non-Maximum Suppression
        img_size (int): Image size for inference
        class_filter (list, optional): List of class names to include
        
    Returns:
        str: Path to the summary video
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Ensure we have valid dimensions
    if width == 0 or height == 0:
        raise ValueError(f"Invalid video dimensions: {width}x{height}")
    
    # Determine output FPS (we want the summary video to play at normal speed)
    output_fps = fps / frame_interval
    if output_fps < 1:
        output_fps = 1  # Ensure minimum 1 FPS
    
    # Choose an appropriate codec based on platform
    import platform
    if platform.system() == 'Windows':
        try:
            # Try H.264 first for better browser compatibility
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            if not output_path.lower().endswith('.mp4'):
                output_path = output_path.rsplit('.', 1)[0] + '.mp4'
        except:
            # Fallback to XVID which is more widely available on Windows
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            if not output_path.lower().endswith('.avi'):
                output_path = output_path.rsplit('.', 1)[0] + '.avi'
    else:
        # For other platforms, try H.264
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        if not output_path.lower().endswith('.mp4'):
            output_path = output_path.rsplit('.', 1)[0] + '.mp4'
    
    # Create video writer
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    if not out.isOpened():
        # Fallback to MJPG which is widely available
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        output_path = output_path.rsplit('.', 1)[0] + '.avi'
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        if not out.isOpened():
            raise ValueError(f"Could not create video writer for: {output_path}")
    
    # Process frames
    frame_idx = 0
    processed_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame
            if frame_idx % frame_interval == 0:
                # Run inference with advanced parameters
                results = model(frame, conf=conf_threshold, iou=iou_threshold, imgsz=img_size)
                
                # If we have a class filter, apply it
                if class_filter:
                    # Create new boxes with only the classes we want
                    filtered_results = []
                    for r in results:
                        filtered_detections = []
                        for box in r.boxes:
                            cls_id = int(box.cls.item())
                            class_name = model.names[cls_id].lower()
                            if class_name in class_filter:
                                filtered_detections.append(box)
                        
                        # Only use results with our filtered classes
                        if filtered_detections:
                            # We would need to modify the results object here to contain only
                            # the filtered detections, but we'll use the plotting function with labels
                            pass
                
                # Plot results on frame with custom settings
                annotated_frame = results[0].plot(conf=conf_threshold, line_width=2, font_size=1.0)
                
                # Add text overlay with detection information
                detections = {}
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls.item())
                        class_name = model.names[cls_id]
                        
                        # Apply class filter if provided
                        if class_filter and class_name.lower() not in class_filter:
                            continue
                            
                        if class_name in detections:
                            detections[class_name] += 1
                        else:
                            detections[class_name] = 1
                
                # Add a header with frame information
                header_text = f"Frame: {frame_idx} | Objects detected: {sum(detections.values())}"
                cv2.putText(annotated_frame, header_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Write to output video
                out.write(annotated_frame)
                processed_count += 1
            
            frame_idx += 1
    except Exception as e:
        print(f"Error processing frame: {e}")
    finally:
        # Release resources
        cap.release()
        out.release()
    
    # Verify that we produced a valid video
    if processed_count == 0:
        raise ValueError("No frames were processed for the summary video")
    
    return output_path

def ensure_web_compatible_video(video_path):
    """
    Ensure a video is web-compatible for Streamlit display
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        str: Path to the web-compatible video
    """
    # Check if ffmpeg is available
    try:
        import subprocess
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        has_ffmpeg = result.returncode == 0
    except:
        has_ffmpeg = False
    
    # If ffmpeg is not available, just return the original path
    if not has_ffmpeg:
        return video_path
    
    # Create a web-compatible version (H.264 MP4)
    web_path = os.path.splitext(video_path)[0] + "_web.mp4"
    
    try:
        cmd = [
            "ffmpeg", 
            "-i", video_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            "-y",  # Overwrite if exists
            web_path
        ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        # Check if the conversion was successful
        if os.path.exists(web_path) and os.path.getsize(web_path) > 0:
            return web_path
        else:
            return video_path
    except Exception as e:
        print(f"Error converting video: {e}")
        return video_path

def extract_key_frames_for_ai(video_path, output_dir, frame_interval=24, max_frames=10):
    """
    Extract key frames from a video for AI analysis and save them to disk
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save extracted frames
        frame_interval (int): Extract every nth frame
        max_frames (int): Maximum number of frames to extract
        
    Returns:
        list: Paths to extracted frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # Calculate frame interval to get a good representation within max_frames
    if frame_count > max_frames * frame_interval:
        # Adjust interval to extract approximately max_frames
        frame_interval = max(1, int(frame_count / max_frames))
    
    # Initialize list to store frame paths
    frame_paths = []
    
    # Read and extract frames
    frame_idx = 0
    extracted_count = 0
    
    while cap.isOpened() and extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every nth frame
        if frame_idx % frame_interval == 0:
            # Save the frame to the output directory
            frame_path = os.path.join(output_dir, f"ai_frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            extracted_count += 1
        
        frame_idx += 1
    
    # Add the last frame if it wasn't included and we haven't reached max_frames
    if extracted_count < max_frames and frame_idx > 0 and (frame_idx - 1) % frame_interval != 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(output_dir, f"ai_frame_last.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
    
    # Release the video capture
    cap.release()
    
    return frame_paths

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