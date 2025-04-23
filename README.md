# Home Video Scanner with AI Room Description

This application allows users to upload videos of their home and uses advanced AI to analyze the rooms and objects within them.

## Features

- **AI Room Description**: Uses OpenAI's GPT-4 Vision to generate detailed descriptions of your space
- **Object Detection**: Uses YOLOv8 to detect and count objects in your home
- **Combined Analysis**: Get both AI descriptions and object detection in one go
- **Summary Videos**: Create annotated videos showing detected objects

## Requirements

- Python 3.8+
- OpenAI API key (for AI Room Description mode)
- Required packages in requirements.txt

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your OpenAI API key:
   - Copy `.env.example` to `.env` and add your API key, or
   - Enter your API key directly in the application

3. Run the application:
   ```
   python run.py
   ```
   
   Or use Streamlit directly:
   ```
   streamlit run app.py
   ```

## Usage

1. Select an analysis mode (AI Room Description, Object Detection, or Combined)
2. Upload a video of your home
3. Click the analysis button
4. View the results:
   - Detailed AI descriptions of your space
   - Lists and charts of detected objects
   - Annotated summary video

## How It Works

- **AI Room Description**: Extracts key frames from your video and uses OpenAI's GPT-4 Vision to generate detailed descriptions of your space.
- **Object Detection**: Uses YOLOv8, a state-of-the-art object detection model, to identify and count objects in your video.
- **Combined Analysis**: Performs both analyses and presents the results together.

## Privacy

All processing is done on your machine. Videos are not sent to external servers except when using OpenAI's API for AI descriptions (which only sends selected video frames). 