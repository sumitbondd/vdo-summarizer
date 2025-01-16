import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai
import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and configure API
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    logger.error("No API key found in environment variables")
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Constants
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
ALLOWED_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo']
ALLOWED_EXTENSIONS = ['mp4', 'mov', 'avi']


class VideoProcessor:
    """Handles video processing operations"""

    @staticmethod
    def validate_video_file(video_file):
        """Validates uploaded video file"""
        if video_file is None:
            raise ValueError("No file was uploaded")

        if video_file.size > MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE / 1024 / 1024}MB limit")

        file_extension = video_file.name.split('.')[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Invalid file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}")

        return True

    @staticmethod
    def create_temp_file(video_file):
        """Creates a temporary file from uploaded video"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            return temp_video.name


class ProgressTracker:
    """Manages progress tracking and status updates"""

    def __init__(self):
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()

    def update_progress(self, stage, progress):
        """Updates progress bar and status text"""
        stages = {
            "upload": ("Uploading video", 0.2),
            "process": ("Processing video", 0.4),
            "analyze": ("Analyzing content", 0.7),
            "complete": ("Completing analysis", 1.0)
        }

        stage_text, stage_progress = stages[stage]
        self.progress_bar.progress(stage_progress)
        self.status_text.text(f"Status: {stage_text}...")
        time.sleep(0.1)


def handle_processing_error(error):
    """Handles various processing errors with specific messages"""
    error_messages = {
        "InvalidFileError": "The uploaded file format is not supported. Please upload MP4, MOV, or AVI files.",
        "ProcessingTimeoutError": "Video processing took too long. Please try with a shorter video.",
        "APIError": "There was an issue connecting to the AI service. Please try again later.",
        "FileCorruptError": "The video file appears to be corrupted. Please upload a different file.",
        "ValueError": str(error)
    }

    error_type = type(error).__name__
    return error_messages.get(error_type, f"An unexpected error occurred: {str(error)}")


@st.cache_resource
def initialize_agent():
    """Initializes the AI agent with Gemini model"""
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )


def main():
    # Page configuration
    st.set_page_config(
        page_title="Multimodal AI Agent- Video Summarizer",
        page_icon="🎥",
        layout="wide"
    )

    st.title("Phidata Video AI Summarizer Agent 🎥🎤🖬")
    st.header("Powered by Gemini 2.0 Flash Exp")

    # Initialize components
    multimodal_Agent = initialize_agent()
    video_processor = VideoProcessor()

    # File uploader
    video_file = st.file_uploader(
        "Upload a video file",
        type=ALLOWED_EXTENSIONS,
        help="Upload a video (MP4, MOV, or AVI) for AI analysis. Maximum size: 100MB"
    )

    if video_file:
        try:
            # Validate and process video
            video_processor.validate_video_file(video_file)
            video_path = video_processor.create_temp_file(video_file)

            # Display video
            st.video(video_path, format="video/mp4", start_time=0)

            # Query input
            user_query = st.text_area(
                "What insights are you seeking from the video?",
                placeholder="Example: 'Summarize the main points discussed in the video' or 'Analyze the emotional tone of the speakers'",
                help="Provide specific questions or insights you want from the video.",
                height=100
            )

            # Analysis button
            if st.button("🔍 Analyze Video", key="analyze_video_button"):
                if not user_query:
                    st.warning("Please enter a question or insight to analyze the video.")
                else:
                    progress_tracker = ProgressTracker()

                    try:
                        # Process video
                        progress_tracker.update_progress("upload", 0.2)
                        processed_video = upload_file(video_path)

                        progress_tracker.update_progress("process", 0.4)
                        while processed_video.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_video = get_file(processed_video.name)

                        # Generate analysis
                        progress_tracker.update_progress("analyze", 0.7)
                        analysis_prompt = f"""
                            Analyze the uploaded video for content and context.
                            Respond to the following query using video insights and supplementary web research:
                            {user_query}

                            Provide a detailed, user-friendly, and actionable response.
                            Include timestamps where relevant.
                            """

                        response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])

                        # Display results
                        progress_tracker.update_progress("complete", 1.0)
                        st.success("Analysis completed successfully!")

                        st.subheader("Analysis Result")
                        st.markdown(response.content)

                        # Add timestamp
                        st.caption(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                    except Exception as error:
                        logger.error(f"Error during analysis: {error}")
                        st.error(handle_processing_error(error))
                    finally:
                        # Cleanup
                        Path(video_path).unlink(missing_ok=True)

        except Exception as error:
            st.error(handle_processing_error(error))
    else:
        st.info("Upload a video file to begin analysis.")

    # Custom styling
    st.markdown(
        """
        <style>
        .stTextArea textarea {
            height: 100px;
        }
        .stButton button {
            width: 200px;
            height: 50px;
            margin: 20px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()