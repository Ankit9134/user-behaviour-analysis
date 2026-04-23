"""
Configuration module for AI Project Hub.
Loads environment variables from .env file and provides them as Python variables.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Determine if we're running in a PyInstaller bundle
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS'.
    APPLICATION_PATH = Path(sys._MEIPASS)
else:
    APPLICATION_PATH = Path(__file__).parent

# Load environment variables from .env file
env_path = APPLICATION_PATH / '.env'
load_dotenv(dotenv_path=env_path)

# Project Paths
FACE_EMOTION_DIR = os.getenv('FACE_EMOTION_DIR', 'Face_Emotion_Recognition')
SENTIMENT_ANALYSIS_DIR = os.getenv('SENTIMENT_ANALYSIS_DIR', 'Sentiment_Analysis')

# Get absolute paths
FACE_EMOTION_PATH = APPLICATION_PATH / FACE_EMOTION_DIR
SENTIMENT_ANALYSIS_PATH = APPLICATION_PATH / SENTIMENT_ANALYSIS_DIR

# Model Files
FACE_EMOTION_MODEL_H5 = os.getenv('FACE_EMOTION_MODEL_H5', 'facialemotionmodel.h5')
FACE_EMOTION_MODEL_JSON = os.getenv('FACE_EMOTION_MODEL_JSON', 'facialemotionmodel.json')

# UI Configuration
WINDOW_WIDTH = int(os.getenv('WINDOW_WIDTH', 1200))
WINDOW_HEIGHT = int(os.getenv('WINDOW_HEIGHT', 700))
WINDOW_SIZE = f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"

# Colors
BG_COLOR = os.getenv('BG_COLOR', '#2c3e50')
TEXT_COLOR = os.getenv('TEXT_COLOR', '#ecf0f1')
ACCENT_COLOR = os.getenv('ACCENT_COLOR', '#3498db')
CARD_BG = os.getenv('CARD_BG', '#34495e')

# Button Colors
PRIMARY_BTN_COLOR = os.getenv('PRIMARY_BTN_COLOR', '#2980b9')
SUCCESS_BTN_COLOR = os.getenv('SUCCESS_BTN_COLOR', '#27ae60')
DANGER_BTN_COLOR = os.getenv('DANGER_BTN_COLOR', '#e74c3c')
NEUTRAL_BTN_COLOR = os.getenv('NEUTRAL_BTN_COLOR', '#7f8c8d')

# Emotion Colors
EMOTION_COLORS = {
    'angry': os.getenv('EMOTION_COLOR_ANGRY', '#e74c3c'),
    'disgust': os.getenv('EMOTION_COLOR_DISGUST', '#8e44ad'),
    'fear': os.getenv('EMOTION_COLOR_FEAR', '#f39c12'),
    'happy': os.getenv('EMOTION_COLOR_HAPPY', '#2ecc71'),
    'neutral': os.getenv('EMOTION_COLOR_NEUTRAL', '#3498db'),
    'sad': os.getenv('EMOTION_COLOR_SAD', '#95a5a6'),
    'surprise': os.getenv('EMOTION_COLOR_SURPRISE', '#1abc9c')
}

# Sentiment Colors
SENTIMENT_COLORS = {
    'positive': os.getenv('SENTIMENT_COLOR_POSITIVE', '#27ae60'),
    'neutral': os.getenv('SENTIMENT_COLOR_NEUTRAL', '#f39c12'),
    'negative': os.getenv('SENTIMENT_COLOR_NEGATIVE', '#e74c3c')
}

# Camera Configuration
DEFAULT_CAMERA_INDEX = int(os.getenv('DEFAULT_CAMERA_INDEX', 0))

# TensorFlow Configuration
TF_CPP_MIN_LOG_LEVEL = os.getenv('TF_CPP_MIN_LOG_LEVEL', '2')

# Application Control
RETURN_TO_MENU_CODE = int(os.getenv('RETURN_TO_MENU_CODE', 42))

# Debug Mode
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() in ('true', '1', 't')

def get_project_path(project_dir):
    """Get the absolute path to a project directory"""
    return APPLICATION_PATH / project_dir

def get_model_path(model_file, project_dir=None):
    """Get the absolute path to a model file"""
    if project_dir:
        return get_project_path(project_dir) / model_file
    return APPLICATION_PATH / model_file

def print_config():
    """Print all configuration variables (for debugging)"""
    print("\n=== AI Project Hub Configuration ===")
    print(f"Application Path: {APPLICATION_PATH}")
    print(f"Face Emotion Path: {FACE_EMOTION_PATH}")
    print(f"Sentiment Analysis Path: {SENTIMENT_ANALYSIS_PATH}")
    print(f"Window Size: {WINDOW_SIZE}")
    print(f"Debug Mode: {DEBUG_MODE}")
    print("====================================\n")

# Set TensorFlow warning level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL

# Print configuration if in debug mode
if DEBUG_MODE:
    print_config()
