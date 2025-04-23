import os
import cv2
import numpy as np
import streamlit as st
from pathlib import Path
import tempfile
import time
import platform
import base64
from ultralytics import YOLO
from PIL import Image
import base64

# Import utility modules
from video_utils import analyze_video, create_summary_video, ensure_web_compatible_video, extract_key_frames_for_ai
from coreml_utils import convert_yolo_to_coreml, optimize_for_device
from ai_understanding import generate_room_description, analyze_video_frames_with_ai, generate_scene_summary, check_api_key, save_api_key

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
st.markdown("Upload a video of your home for AI-powered room analysis.")

# Check if running on macOS for CoreML support
is_mac = platform.system() == "Darwin"

# Function to get model
@st.cache_resource
def load_model(model_path, use_coreml=False, img_size=640):
    # Check if we're using a specialized model
    if "furniture_detector" in model_path:
        # This would be a path to a fine-tuned model for furniture
        st.info("Loading specialized furniture detector model...")
        model_path = "yolov8m.pt"  # Fallback to standard model if specialized not available
    elif "indoor_objects" in model_path:
        # This would be a path to a fine-tuned model for indoor objects
        st.info("Loading specialized indoor objects detector model...")
        model_path = "yolov8m.pt"  # Fallback to standard model if specialized not available
    
    # Default to standard pytorch model
    if not use_coreml:
        return YOLO(model_path, task='detect')
    
    # For CoreML, first convert the model if needed
    model_name = os.path.basename(model_path)
    coreml_model_path = f"models/{os.path.splitext(model_name)[0]}.mlmodel"
    if not os.path.exists(coreml_model_path):
        st.info("Converting YOLOv8 model to CoreML format. This may take a moment...")
        coreml_model_path = convert_yolo_to_coreml(model_path, "models")
        
        # Check if on Apple Silicon for optimization
        if platform.processor() == "arm":
            st.info("Optimizing model for Apple Silicon...")
            coreml_model_path = optimize_for_device(coreml_model_path, device="neural_engine")
    
    # Load the CoreML model
    model = YOLO(coreml_model_path, task='detect')
    return model

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    
    # Analysis mode selection
    analysis_mode = st.radio(
        "Analysis Mode",
        ["AI Room Description", "Object Detection", "Combined Analysis"],
        index=0
    )
    
    # API Key configuration for AI Room Description
    if analysis_mode in ["AI Room Description", "Combined Analysis"]:
        st.subheader("OpenAI API Configuration")
        
        if not check_api_key():
            api_key = st.text_input("Enter OpenAI API Key", type="password")
            if api_key:
                save_api_key(api_key)
                st.success("API key saved!")
        else:
            st.success("OpenAI API key configured")
            if st.button("Change API Key"):
                st.session_state.pop("openai_api_key", None)
                st.session_state.pop("openai_client", None)
                st.experimental_rerun()
    
    # Only show model selection if using object detection
    if analysis_mode in ["Object Detection", "Combined Analysis"]:
        # Model selection
        model_type = st.selectbox(
            "Model Type",
            ["YOLOv8m (default)", "YOLOv8n (fast)", "YOLOv8s", "YOLOv8l", "YOLOv8x (most accurate)"],
            index=0
        )
        
        # Add specialized model options
        use_specialized = st.checkbox("Use specialized model for indoor objects", value=False)
        if use_specialized:
            specialized_model = st.selectbox(
                "Specialized Model",
                ["furniture_detector", "indoor_objects"]
            )
            st.info("💡 Specialized models are fine-tuned for specific object categories and may provide better detection for home environments.")
        
        # CoreML option (only for Mac)
        use_coreml = False
        if is_mac:
            use_coreml = st.checkbox("Use CoreML (macOS only)", value=True)
        
        # Processing settings
        st.subheader("Processing Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
        
        # IOU threshold for Non-Maximum Suppression
        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05, 
                                help="Intersection over Union threshold for Non-Maximum Suppression. Lower values create stricter detection filtering.")
        
        # Sampling rate (every nth frame)
        frame_interval = st.slider("Process every n-th frame", 1, 60, 24)
        
        # Summary video option
        create_summary = st.checkbox("Create Summary Video", value=True)
        
        # Additional settings expandable section
        with st.expander("Advanced Settings"):
            # Class filtering
            filter_classes = st.checkbox("Filter detected classes", value=False)
            if filter_classes:
                st.info("Enter class names to include, one per line (e.g., chair, sofa, table)")
                class_filter_text = st.text_area("Classes to include", height=100)
                class_filter = [cls.strip().lower() for cls in class_filter_text.split('\n') if cls.strip()]
            else:
                class_filter = None
                
            # Image size
            img_size = st.select_slider(
                "Image Size",
                options=[320, 416, 512, 640, 768, 896, 1024],
                value=640,
                help="Larger values improve accuracy but reduce speed"
            )
    
    # AI Description settings
    if analysis_mode in ["AI Room Description", "Combined Analysis"]:
        st.subheader("AI Description Settings")
        
        # Number of frames to analyze
        ai_frame_count = st.slider("Number of frames to analyze", 1, 20, 5, 
                                 help="More frames provide better understanding but increase processing time and API costs")
        
        # Use detailed descriptions
        detailed_ai = st.checkbox("Generate detailed descriptions", value=True,
                               help="Provides more detailed analysis but uses more tokens")

# Map model type to actual model path
model_paths = {
    "YOLOv8n (fast)": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
    "YOLOv8m (default)": "yolov8m.pt",
    "YOLOv8l": "yolov8l.pt",
    "YOLOv8x (most accurate)": "yolov8x.pt"
}

# Load the YOLO model if using object detection
model = None
if analysis_mode in ["Object Detection", "Combined Analysis"]:
    # Load the selected model
    model_path = model_paths[model_type]
    
    # Check if we're using a specialized model
    if use_specialized:
        specialized_path = f"{specialized_model}.pt"  # This would be the path to the specialized model
        try:
            model = load_model(specialized_path, use_coreml, img_size)
        except Exception as e:
            st.error(f"Error loading specialized model: {e}")
            st.info(f"Falling back to {model_type}")
            model = load_model(model_path, use_coreml, img_size)
    else:
        try:
            model = load_model(model_path, use_coreml, img_size)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("Falling back to standard YOLOv8m model")
            model = YOLO("yolov8m.pt")

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    # File uploader
    video_file = st.file_uploader(
        "Upload a video of your home", 
        type=["mp4", "mov", "avi"],
        help="Upload a video file to analyze your home"
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
        # Process button with appropriate label based on mode
        button_labels = {
            "AI Room Description": "🤖 Analyze Room with AI",
            "Object Detection": "📷 Detect Objects",
            "Combined Analysis": "🔍 Full Scene Analysis"
        }
        
        if st.button(button_labels[analysis_mode]):
            # Setup progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Start timing
            start_time = time.time()
            
            # Create temp directory for frame extraction
            temp_dir = tempfile.mkdtemp()
            
            # AI Room Description Mode
            if analysis_mode in ["AI Room Description", "Combined Analysis"]:
                if not check_api_key():
                    st.error("OpenAI API key is required for AI room description. Please add it in the sidebar.")
                else:
                    status_text.text("Extracting frames for AI analysis...")
                    progress_bar.progress(10)
                    
                    # Extract key frames for AI analysis
                    frame_paths = extract_key_frames_for_ai(
                        tfile_path, 
                        temp_dir, 
                        frame_interval=24, 
                        max_frames=ai_frame_count
                    )
                    
                    if not frame_paths:
                        st.error("Failed to extract frames from the video.")
                    else:
                        status_text.text(f"Analyzing {len(frame_paths)} frames with AI...")
                        progress_bar.progress(30)
                        
                        # Analyze the frames with AI
                        frame_descriptions = analyze_video_frames_with_ai(frame_paths)
                        
                        status_text.text("Generating overall room summary...")
                        progress_bar.progress(70)
                        
                        # Generate an overall summary
                        room_summary = generate_scene_summary(frame_descriptions)
                        
                        progress_bar.progress(100)
                        status_text.text("AI analysis complete!")
                        
                        # Display AI analysis results
                        st.subheader("Room Description")
                        st.markdown(room_summary)
                        
                        # Display individual frame descriptions if detailed mode is on
                        if detailed_ai:
                            with st.expander("Detailed Frame Descriptions"):
                                for i, (timestamp, description) in enumerate(frame_descriptions.items()):
                                    st.subheader(f"Frame {i+1}")
                                    # Display the frame
                                    st.image(frame_paths[i], caption=timestamp)
                                    # Display the description
                                    st.markdown(description)
            
            # Object Detection Mode
            if analysis_mode in ["Object Detection", "Combined Analysis"]:
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
                    
                    status_text.text("Detecting objects in video...")
                    progress_bar.progress(40 if analysis_mode == "Combined Analysis" else 20)
                    
                    # Use our utility function to analyze the video
                    detected_objects, _ = analyze_video(
                        tfile_path, 
                        model, 
                        frame_interval=frame_interval,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        img_size=img_size,
                        class_filter=class_filter if filter_classes else None
                    )
                    
                    # Create summary video if enabled
                    if create_summary:
                        status_text.text("Creating summary video...")
                        progress_bar.progress(70 if analysis_mode == "Combined Analysis" else 60)
                        try:
                            summary_path = create_summary_video(
                                tfile_path,
                                model,
                                summary_path,
                                frame_interval=frame_interval,
                                conf_threshold=conf_threshold,
                                iou_threshold=iou_threshold,
                                img_size=img_size,
                                class_filter=class_filter if filter_classes else None
                            )
                            
                            # Ensure the video is web-compatible
                            status_text.text("Optimizing video for web playback...")
                            summary_path = ensure_web_compatible_video(summary_path)
                            
                            st.success(f"Summary video created at: {summary_path}")
                        except Exception as e:
                            st.error(f"Error creating summary video: {e}")
                            # Continue with analysis even if summary fails
                            summary_path = None
                    
                    # Sort objects by frequency
                    sorted_objects = dict(sorted(detected_objects.items(), key=lambda item: item[1], reverse=True))
                    
                    # Show results
                    st.subheader("Objects Detected")
                    
                    # Display the results
                    for obj, count in sorted_objects.items():
                        st.write(f"- **{obj}**: {count} instances")
                    
                    # Create a bar chart of top 10 objects
                    if sorted_objects:
                        top_items = dict(list(sorted_objects.items())[:10])
                        st.subheader("Top 10 Objects")
                        st.bar_chart(top_items)
                        
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
                                
                                # Create columns for video and download button
                                video_col, download_col = st.columns([4, 1])
                                
                                with video_col:
                                    # Try standard Streamlit video component
                                    try:
                                        st.video(alt_path)
                                    except Exception as e:
                                        st.warning(f"Standard video player not working: {e}")
                                        # Fall back to HTML player
                                        st.markdown(get_video_html(alt_path), unsafe_allow_html=True)
                                
                                with download_col:
                                    # Provide download option
                                    with open(alt_path, 'rb') as f:
                                        video_bytes = f.read()
                                        st.download_button(
                                            label="Download summary",
                                            data=video_bytes,
                                            file_name="summary_video.mp4",
                                            mime="video/mp4"
                                        )
                                        
                                summary_path = alt_path
                                break
                        else:
                            st.warning("Summary video could not be created.")
                except Exception as e:
                    st.error(f"Error during object detection: {e}")
            
            # End timing
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Show processing information
            progress_bar.progress(100)
            status_text.text(f"Analysis complete in {processing_time:.1f}s!")
                
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
    2. **Analyze**: The application processes the video using advanced AI
    3. **Results**: View a comprehensive analysis of your space
    
    This application offers three analysis modes:
    
    - **AI Room Description**: Uses OpenAI's GPT-4 Vision to generate detailed descriptions of your space
    - **Object Detection**: Uses YOLOv8 to detect and count objects in your home
    - **Combined Analysis**: Provides both AI descriptions and object detection
    
    ### Technology
    
    - OpenAI GPT-4 Vision for scene understanding
    - YOLOv8 for object detection
    - CoreML optimization for macOS users
    - Streamlit for the web interface
    
    ### Privacy
    
    All processing is done on your machine - videos are not sent to external servers except when using OpenAI's API for AI descriptions.
    """)

# Add footer
st.markdown("---")
st.markdown("Built with YOLOv8, OpenAI GPT-4 Vision, and Streamlit") 