# Hey Rodea - AI Speech Coach & Transcriber

An intelligent speech analysis and coaching application that combines automatic speech recognition with personalized feedback to help users improve their speaking skills.

## Overview

Hey Rodea is a comprehensive solution for speech analysis and improvement. The application leverages state-of-the-art automatic speech recognition (ASR) technology to transcribe audio content and provides detailed coaching feedback on speaking performance metrics including clarity, confidence, engagement, pace, and filler word usage.

## Features

### Speech Recognition
- Automatic transcription of audio files in MP3 and WAV formats
- Support for multiple audio sources including file uploads and live recordings
- Word-level timestamp alignment for precise timing information
- Integration with industry-standard ASR models

### Speech Analysis
The application analyzes multiple dimensions of speech performance:

- **Clarity Score**: Evaluates pronunciation quality and articulation
- **Confidence Score**: Assesses vocal delivery and presence
- **Engagement Score**: Measures listener interest and speaking dynamics
- **Words Per Minute (WPM)**: Tracks speaking pace and rhythm
- **Pause Detection**: Identifies and analyzes strategic pauses
- **Filler Word Analysis**: Detects and quantifies filler words (um, uh, like, etc.)

### Actionable Feedback
- Personalized improvement recommendations based on analysis results
- Identification of speaking strengths
- Specific areas for improvement with targeted suggestions
- Session history for tracking progress over time

### User Interface
- Interactive web-based interface built with Streamlit
- Visual performance metrics with donut charts
- Detailed transcript view with timing information
- Session management and history tracking
- Responsive design for desktop and mobile devices

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/themrandroid/hey-rodea.git
cd hey-rodea
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure any required API keys or environment variables as needed for your ASR backend.

## Usage

### Running the Application

Start the application using Streamlit:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Basic Workflow

1. Upload or record an audio file (MP3 or WAV format)
2. The application processes the audio and generates a transcript
3. Speech analysis metrics are calculated automatically
4. Review your performance scores and personalized feedback
5. View your session history to track improvements over time

### File Structure

The project includes:
- `app.py` - Main Streamlit application entry point
- `pipeline.py` - Core speech processing and analysis pipeline
- `app_design.py` - UI components and styling
- `data_collection.ipynb` - Jupyter notebook for data collection workflow
- `pipeline.ipynb` - Analysis pipeline development notebook
- `asr_json` - Cached ASR output in JSON format with word-level timings
- `librispeech_subset_wav` - Sample audio files for testing
- `processed` - Processed analysis results
- `uploads` - User-uploaded audio files

## Data

The project includes sample datasets:

- **LibriSpeech Subset**: High-quality English speech samples for testing and development
- **Common Voice Data**: Multilingual speech samples for robust model training
- **Metadata**: CSV files containing audio metadata and transcriptions

See `librispeech_metadata.csv` and `metadata_commonvoice.csv` for dataset documentation.

## Technical Architecture

### Speech Recognition
The application uses advanced ASR models to convert audio to text with word-level timing information stored in JSON format for downstream analysis.

### Analysis Pipeline
The processing pipeline implements:
- Audio preprocessing and normalization
- Speech segmentation and pause detection
- Scoring algorithms for clarity, confidence, and engagement metrics
- Filler word detection and classification
- Report generation with actionable feedback

### Data Storage
- Audio files stored in WAV format with consistent 16kHz sampling rate
- ASR results cached in JSON format for quick retrieval
- Session data and analysis results stored for historical tracking

## Live Demo

Try the application here: [hey-rodea.streamlit.app](https://hey-rodea.streamlit.app/)

## Performance Metrics

### Supported Languages
- English (primary)
- Extensible to additional languages through model configuration

### Audio Quality
- Optimal quality: 16kHz sample rate, mono channel
- File size: Up to 100MB supported
- Formats: MP3, WAV

### Processing Time
Typical processing times depend on audio length and system resources. Real-time processing is supported for live recordings.

## Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
```

### Running Notebooks

Analysis and data collection workflows are available in Jupyter notebooks:

```bash
jupyter notebook pipeline.ipynb
jupyter notebook data_collection.ipynb
```

## Performance Considerations

- First-time model loading may take 30-60 seconds
- Subsequent requests use cached models for faster processing
- Batch processing available for multiple files