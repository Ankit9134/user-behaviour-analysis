# AI Project Hub

A comprehensive application that combines multiple AI projects into a single, user-friendly interface. Currently includes Face Emotion Recognition and Sentiment Analysis capabilities.

## Features

- **Face Emotion Recognition**: Detect and analyze emotions in real-time using your webcam
- **Sentiment Analysis**: Analyze the sentiment of text input
- **Unified Interface**: Easy-to-use main menu for accessing all projects
- **Configurable**: Easily customize the application through environment variables

## Screenshots

*[Add screenshots of your application here]*

## Requirements

- Python 3.8 or higher
- Webcam (for Face Emotion Recognition)
- Dependencies listed in `requirements.txt`

## Installation

### Setting Up a Virtual Environment

It's recommended to use a virtual environment to avoid conflicts with other Python projects.

#### Windows

```bash
# Navigate to the project directory
cd "path/to/AI Project Hub"

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### macOS/Linux

```bash
# Navigate to the project directory
cd "path/to/AI Project Hub"

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Additional Setup

For the Face Emotion Recognition project, you need to download the model file:

1. Make sure the `facialemotionmodel.h5` file is in the `Face_Emotion_Recognition` directory
2. If you don't have this file, you can train your own model using the provided Jupyter notebook in the `Face_Emotion_Recognition` directory

For the Sentiment Analysis project, you need to download NLTK data:

```bash
# Start Python
python

# In the Python interpreter
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Exit Python
exit()
```

## Usage

### Running the Application

After activating your virtual environment, run the main application:

```bash
python main.py
```

This will open the main menu where you can select which project to launch.

### Face Emotion Recognition

1. Select "Face Emotion Recognition" from the main menu
2. Click "Start Camera" to begin detecting emotions
3. The application will display the detected emotion and update statistics in real-time
4. Click "Reset Stats" to clear the emotion statistics
5. Click "Return to Main Menu" to go back to the main menu

### Sentiment Analysis

1. Select "Sentiment Analysis" from the main menu
2. Enter text in the input area
3. Click "Analyze Sentiment" to analyze the sentiment of the text
4. The application will display the sentiment analysis results
5. Click "Clear" to clear the input and results
6. Click "Return to Main Menu" to go back to the main menu

## Configuration

The application can be configured using environment variables in the `.env` file:

```
# Example .env file
WINDOW_WIDTH=1200
WINDOW_HEIGHT=700
DEBUG_MODE=False
```

See the `.env` file for all available configuration options.

## Project Structure

```
AI Project Hub/
├── .env                           # Environment variables
├── config.py                      # Configuration module
├── main.py                        # Main application entry point
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── Face_Emotion_Recognition/      # Face Emotion Recognition project
│   ├── MainRealtimeEmotion.py     # Main script for face emotion recognition
│   ├── facialemotionmodel.h5      # Pre-trained model
│   └── trainmodel.ipynb           # Notebook for training the model
└── Sentiment_Analysis/            # Sentiment Analysis project
    ├── analysis.py                # Main script for sentiment analysis
    ├── emotions.txt               # Emotion words dictionary
    └── main_nltk.py               # Alternative NLTK-based script

```

## Troubleshooting

### Common Issues

1. **Missing Model File**: Ensure `facialemotionmodel.h5` is in the `Face_Emotion_Recognition` directory
2. **Webcam Not Working**: Check your webcam connection and permissions
3. **NLTK Data Missing**: Run the NLTK download commands in the Additional Setup section

### Error Messages

- **"Failed to run Face Emotion Recognition"**: Check that all dependencies are installed and the model file exists
- **"Could not open webcam"**: Check your webcam connection and permissions
- **"Error loading model"**: Ensure the model file is in the correct location and format

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Acknowledgments

- TensorFlow and Keras for the deep learning framework
- OpenCV for computer vision capabilities
- NLTK and VADER for sentiment analysis
- All other open-source libraries used in this project
