import os
import base64
import json
import requests
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
def get_openai_client():
    """Get OpenAI client from API key in environment or session state"""
    if "openai_client" in st.session_state:
        return st.session_state.openai_client
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    
    client = OpenAI(api_key=api_key)
    st.session_state.openai_client = client
    return client

def encode_image_to_base64(image_path):
    """Convert an image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_room_description(image_path, model="gpt-4-vision-preview"):
    """
    Generate a description of a room from an image using OpenAI's vision model
    
    Args:
        image_path (str): Path to the image file
        model (str): OpenAI model to use
        
    Returns:
        str: Description of the room
    """
    client = get_openai_client()
    if not client:
        return "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
    
    try:
        # Encode image to base64
        base64_image = encode_image_to_base64(image_path)
        
        # Create a message with the image
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant specialized in describing rooms and indoor spaces. "
                              "Focus on furniture, decor, layout, color schemes, and notable objects. "
                              "For each image, provide a comprehensive description of what's in the room."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this room in detail, including all the objects, furniture, and features you can see."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=500
        )
        
        # Extract the description from the response
        description = response.choices[0].message.content
        return description
    
    except Exception as e:
        return f"Error generating description: {str(e)}"

def analyze_video_frames_with_ai(frames_paths, sample_rate=5):
    """
    Analyze key frames from a video using AI vision model
    
    Args:
        frames_paths (list): List of paths to video frames
        sample_rate (int): Analyze every nth frame to reduce API costs
        
    Returns:
        dict: Frame timestamps and their descriptions
    """
    results = {}
    total_frames = len(frames_paths)
    
    for i, frame_path in enumerate(frames_paths):
        # Only analyze every nth frame to save API costs
        if i % sample_rate != 0 and i != 0 and i != total_frames - 1:
            continue
            
        # Generate description for this frame
        description = generate_room_description(frame_path)
        
        # Store the result
        timestamp = f"Frame {i}"
        results[timestamp] = description
        
    return results

def generate_scene_summary(frame_descriptions):
    """
    Generate an overall summary of the scene from individual frame descriptions
    
    Args:
        frame_descriptions (dict): Dictionary of frame timestamps and descriptions
        
    Returns:
        str: Overall summary of the scene
    """
    client = get_openai_client()
    if not client:
        return "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
    
    try:
        # Format the frame descriptions as a string
        frames_text = "\n\n".join([f"{timestamp}:\n{description}" 
                                  for timestamp, description in frame_descriptions.items()])
        
        # Create a message requesting a summary
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes room descriptions. "
                              "You will be given descriptions of frames from a video of a house or room. "
                              "Create a comprehensive summary of what's in the house/room based on all frames. "
                              "Identify consistent objects, furniture, and features across frames. "
                              "Describe the overall layout, style, and purpose of the space."
                },
                {
                    "role": "user",
                    "content": f"Here are descriptions of several frames from a video of a house or room:\n\n{frames_text}\n\n"
                              f"Please provide a comprehensive summary of the house/room shown in these frames."
                }
            ],
            max_tokens=1000
        )
        
        # Extract the summary from the response
        summary = response.choices[0].message.content
        return summary
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def check_api_key():
    """Check if OpenAI API key is configured"""
    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
    return api_key is not None and len(api_key) > 0

def save_api_key(api_key):
    """Save API key to session state"""
    st.session_state.openai_api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    st.session_state.openai_client = OpenAI(api_key=api_key)
    return True 