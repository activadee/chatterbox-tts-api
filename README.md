---
title: Chatterbox TTS API
emoji: 🗣️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: mit
tags:
  - text-to-speech
  - voice-cloning
  - audio
  - ai
  - chatterbox
---

# Chatterbox TTS API

A Gradio-based web interface and API for Chatterbox TTS (Text-to-Speech) with voice cloning and dialogue generation capabilities.

## Features

- **Text-to-Speech**: Convert text to natural-sounding speech with voice cloning
- **Voice Cloning**: Clone voices from audio samples  
- **Dialogue Generation**: Generate multi-speaker conversations with different voices
- **Web Interface**: Easy-to-use Gradio interface for interactive use
- **API Access**: RESTful API endpoints for programmatic access

## Access

### Hugging Face Space
🔗 **Web Interface**: https://activadee-tts-api.hf.space/
🔗 **API Access**: https://activadee-tts-api.hf.space/gradio_api/docs

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/activadee/chatterbox-tts-api.git
cd chatterbox-tts-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The interface will be available at `http://localhost:7860`

## API Endpoints

The API is accessible via Gradio's built-in endpoints once `show_api=True` is enabled:

### Single Voice TTS
```bash
POST /gradio_api/call/speech_ui_to_json
```

**Parameters:**
- `text` (string): Text to convert to speech
- `voice_file` (file, optional): Audio file for voice cloning
- `exaggeration` (float): Voice exaggeration level (0.0-1.0)
- `cfg_weight` (float): Configuration weight (0.0-1.0)

### Dialogue Generation
```bash
POST /gradio_api/call/dialogue_ui_to_json
```

**Parameters:**
- `text` (string): Dialogue text
- `voice1_file` (file, optional): Voice sample for first speaker
- `voice2_file` (file, optional): Voice sample for second speaker
- `speaker_segments` (string): JSON string with speaker segments

## Usage Examples

### Using Gradio Client (Recommended)
```python
from gradio_client import Client

# Connect to the space
client = Client("https://activadee-tts-api.hf.space/")

# Single voice TTS
result = client.predict(
    text="Hello, this is a test of Chatterbox TTS.",
    voice_file=None,  # or path to voice file for cloning
    exaggeration=0.5,
    cfg_weight=0.5,
    api_name="/speech_ui_to_json"
)
print(f"Generated audio: {result}")

# Dialogue generation
dialogue_result = client.predict(
    text="Hello there! How are you doing today?",
    voice1_file=None,
    voice2_file=None, 
    speaker_segments='[{"speaker":"SPEAKER_00","text":"Hello there!"},{"speaker":"SPEAKER_01","text":"How are you doing today?"}]',
    api_name="/dialogue_ui_to_json"
)
print(f"Generated dialogue: {dialogue_result}")
```

### Using REST API (Direct)
```python
import requests

# Single voice TTS
response = requests.post(
    "https://activadee-tts-api.hf.space/gradio_api/call/speech_ui_to_json",
    json={
        "data": [
            "Hello, this is a test of Chatterbox TTS.",  # text
            None,  # voice_file
            0.5,   # exaggeration  
            0.5    # cfg_weight
        ],
        "session_hash": "demo_session"
    }
)

# Check for event_id and poll for results
print(response.json())
```

### Using n8n Node
1. Install the n8n-nodes-gradio-client package
2. Add the Gradio Client node to your workflow
3. Configure:
   - **Space URL**: `https://activadee-tts-api.hf.space/`
   - **API Selection**: Auto-detect from Space
   - **API Name**: Choose from `/speech_ui_to_json` or `/dialogue_ui_to_json`
   - **Input Parameters**: `["Your text here", null, 0.5, 0.5]`

## API Documentation

Once the Space is running with `show_api=True`, interactive API documentation is available at:
- **Gradio API Docs**: https://activadee-tts-api.hf.space/gradio_api/docs
- **Local API Docs**: `http://localhost:7860/gradio_api/docs` (when running locally)

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with [Chatterbox TTS](https://github.com/jasonppy/chatterbox)
- Speaker diarization powered by [pyannote.audio](https://github.com/pyannote/pyannote-audio)