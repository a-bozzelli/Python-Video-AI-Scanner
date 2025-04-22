import os
import cv2
import numpy as np
import streamlit as st
from pathlib import Path
import tempfile
import time
import platform
from ultralytics import YOLO
from PIL import Image
import base64

# Import utility modules
from video_utils import analyze_video, create_summary_video, ensure_web_compatible_video
from coreml_utils import convert_yolo_to_coreml, optimize_for_device

# Helper function for video display
def get_video_html(video_path):
    """Create an HTML5 video player for the given video path"""
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    video_file.close()
    
    # Get the mime type based on file extension
    ext = os.path.splitext(video_path)[1].lower()
    mime_type = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime'
    }.get(ext, 'video/mp4')
    
    # Encode the video bytes as base64
    b64 = base64.b64encode(video_bytes).decode()
    
    # Create an HTML5 video player
    video_html = f"""
    <video width="100%" controls autoplay>
        <source src="data:{mime_type};base64,{b64}" type="{mime_type}">
        Your browser does not support the video tag.
    </video>
    """
    return video_html

# Set page configuration
st.set_page_config(
    page_title="Home Video Scanner",
    page_icon="🏠",
    layout="wide"
)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Application title and description
st.title("Home Video Scanner")
st.markdown("Upload a video of your home to identify objects in each room.")

# Check if running on macOS for CoreML support
is_mac = platform.system() == "Darwin"

# Function to get model
@st.cache_resource
def load_model(use_coreml=False):
    # Default to standard pytorch model
    if not use_coreml:
        return YOLO("yolov8n.pt")
    
    # For CoreML, first convert the model if needed
    coreml_model_path = "models/yolov8n.mlmodel"
    if not os.path.exists(coreml_model_path):
        st.info("Converting YOLOv8 model to CoreML format. This may take a moment...")
        coreml_model_path = convert_yolo_to_coreml("yolov8n.pt", "models")
        
        # Check if on Apple Silicon for optimization
        if platform.processor() == "arm":
            st.info("Optimizing model for Apple Silicon...")
            coreml_model_path = optimize_for_device(coreml_model_path, device="neural_engine")
    
    # Load the CoreML model
    model = YOLO(coreml_model_path)
    return model

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    
    # Model selection
    model_type = st.selectbox(
        "Model Type",
        ["YOLOv8n (default)", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"],
        index=0
    )
    
    # CoreML option (only for Mac)
    use_coreml = False
    if is_mac:
        use_coreml = st.checkbox("Use CoreML (macOS only)", value=True)
    
    # Processing settings
    st.subheader("Processing Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Sampling rate (every nth frame)
    frame_interval = st.slider("Process every n-th frame", 1, 60, 24)
    
    # Summary video option
    create_summary = st.checkbox("Create Summary Video", value=True)

# Map model type to actual model path
model_paths = {
    "YOLOv8n (default)": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
    "YOLOv8m": "yolov8m.pt",
    "YOLOv8l": "yolov8l.pt",
    "YOLOv8x": "yolov8x.pt"
}

# Load the selected model
model_path = model_paths[model_type]
try:
    model = load_model(use_coreml)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Falling back to standard YOLOv8n model")
    model = YOLO("yolov8n.pt")

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    # File uploader
    video_file = st.file_uploader(
        "Upload a video of your home", 
        type=["mp4", "mov", "avi"],
        help="Upload a video file to analyze the objects in your home"
    )

    if video_file is not None:
        # Create a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(video_file.read())
        tfile_path = tfile.name
        
        # Display the uploaded video
        video_display_col, download_col = st.columns([4, 1])
        with video_display_col:
            try:
                st.video(tfile_path)
            except Exception as e:
                st.error(f"Video player error: {e}")
                st.markdown(get_video_html(tfile_path), unsafe_allow_html=True)
        
        with download_col:
            # Provide a download link in case the video doesn't play in browser
            with open(tfile_path, 'rb') as f:
                video_bytes = f.read()
                st.download_button(
                    label="Download video",
                    data=video_bytes,
                    file_name="uploaded_video.mp4",
                    mime="video/mp4"
                )

with col2:
    if video_file is not None:
        # Process button
        if st.button("📷 Scan for Objects"):
            # Setup progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Start timing
            start_time = time.time()
            
            status_text.text("Processing video...")
            
            # Create a placeholder for the current frame
            frame_placeholder = st.empty()
            
            try:
                # Process the video
                cap = cv2.VideoCapture(tfile_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps
                cap.release()
                
                # Set the save path for summary video
                summary_path = None
                if create_summary:
                    # Create separate temp directory for summary video that won't be auto-deleted
                    summary_dir = tempfile.mkdtemp()
                    summary_path = os.path.join(summary_dir, "summary.mp4")
                
                # Use our utility function to analyze the video
                detected_objects, _ = analyze_video(
                    tfile_path, 
                    model, 
                    frame_interval=frame_interval,
                    conf_threshold=conf_threshold
                )
                
                # Create summary video if enabled
                if create_summary:
                    status_text.text("Creating summary video...")
                    try:
                        summary_path = create_summary_video(
                            tfile_path,
                            model,
                            summary_path,
                            frame_interval=frame_interval,
                            conf_threshold=conf_threshold
                        )
                        
                        # Ensure the video is web-compatible
                        status_text.text("Optimizing video for web playback...")
                        summary_path = ensure_web_compatible_video(summary_path)
                        
                        st.success(f"Summary video created at: {summary_path}")
                    except Exception as e:
                        st.error(f"Error creating summary video: {e}")
                        # Continue with analysis even if summary fails
                        summary_path = None
                
                # End timing
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Sort objects by frequency
                sorted_objects = dict(sorted(detected_objects.items(), key=lambda item: item[1], reverse=True))
                
                # Show processing information
                status_text.text(f"Scan complete! Processed {duration:.1f}s video in {processing_time:.1f}s")
                
                # Show summary video if created
                if summary_path and os.path.exists(summary_path):
                    st.subheader("Summary Video")
                    
                    # Store path for cleanup later
                    st.session_state.summary_video_path = summary_path
                    
                    # Create columns for video and download button
                    video_col, download_col = st.columns([4, 1])
                    
                    with video_col:
                        # First try standard Streamlit video component
                        try:
                            st.video(summary_path)
                        except Exception as e:
                            st.warning(f"Standard video player not working: {e}")
                            # Fall back to HTML player
                            st.markdown(get_video_html(summary_path), unsafe_allow_html=True)
                            
                    with download_col:
                        # Provide download option
                        with open(summary_path, 'rb') as f:
                            video_bytes = f.read()
                            st.download_button(
                                label="Download summary",
                                data=video_bytes,
                                file_name="summary_video.mp4",
                                mime="video/mp4"
                            )
                
                elif summary_path:
                    # Check if the file extension changed (e.g., from .mp4 to .avi)
                    possible_extensions = ['.avi', '.mp4', '.mov']
                    for ext in possible_extensions:
                        alt_path = os.path.splitext(summary_path)[0] + ext
                        if os.path.exists(alt_path):
                            st.subheader("Summary Video")
                            
                            # Store path for cleanup later
                            st.session_state.summary_video_path = alt_path
                            
                            # Try standard Streamlit video component
                            try:
                                st.video(alt_path)
                            except Exception as e:
                                st.warning(f"Standard video player not working: {e}")
                                # Fall back to HTML player
                                st.markdown(get_video_html(alt_path), unsafe_allow_html=True)
                                
                            summary_path = alt_path
                            break
                    else:
                        st.warning("Summary video could not be created.")
                
                # Clear progress bar when done
                progress_bar.empty()
                
                # Show results
                st.subheader("Objects Detected in Your Home")
                
                # Display the results
                for obj, count in sorted_objects.items():
                    st.write(f"- **{obj}**: {count} instances")
                
                # Create a bar chart of top 10 objects
                if sorted_objects:
                    top_items = dict(list(sorted_objects.items())[:10])
                    st.subheader("Top 10 Objects")
                    st.bar_chart(top_items)
                
            except Exception as e:
                st.error(f"Error processing video: {e}")
            finally:
                # Clean up the temporary file but only for the uploaded video
                # Keep the summary video available
                try:
                    os.unlink(tfile_path)
                except:
                    pass

# Information section
st.markdown("---")
with st.expander("About Home Video Scanner"):
    st.markdown("""
    ### How it works
    
    1. **Upload**: Upload a video of your home or room
    2. **Scan**: The application processes the video using YOLOv8, a state-of-the-art object detection model
    3. **Results**: View a list of detected objects and their counts
    
    ### Technology
    
    - YOLOv8 for object detection
    - CoreML optimization for macOS users
    - Streamlit for the web interface
    
    ### Privacy
    
    All processing is done locally - your videos are not sent to external servers.
    """)

# Add footer
st.markdown("---")
st.markdown("Built with YOLOv8 and Streamlit") 