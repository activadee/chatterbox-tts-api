# Chatterbox TTS API

A standalone REST API service for Chatterbox TTS (Text-to-Speech) with voice cloning, speaker diarization, and dialogue generation capabilities.

## Features

- **Text-to-Speech**: Convert text to natural-sounding speech
- **Voice Cloning**: Clone voices from audio samples
- **Speaker Diarization**: Identify and separate different speakers in audio
- **Dialogue Generation**: Generate multi-speaker dialogues with different voices
- **Video Pipeline**: Complete pipeline for generating dialogue videos (audio only, video generation via FFmpeg)

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- Docker (optional)

## Installation

### Using Docker

```bash
docker-compose up -d
```

### Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/activadee/chatterbox-tts-api.git
cd chatterbox-tts-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```bash
GET /health
```

### Text-to-Speech
```bash
POST /tts
```

Parameters:
- `text` (string, required): Text to convert to speech
- `exaggeration` (float, default: 0.5): Voice exaggeration level
- `cfg_weight` (float, default: 0.5): Configuration weight
- `return_format` (string, default: "wav"): Audio format (wav/mp3)
- `voice_sample` (file, optional): Audio file for voice cloning

### Speaker Diarization
```bash
POST /speaker-diarization
```

Parameters:
- `audio_file` (file, required): Audio file to analyze
- `min_speakers` (int, optional): Minimum number of speakers
- `max_speakers` (int, optional): Maximum number of speakers

### Generate Dialogue Audio
```bash
POST /generate-dialogue-audio
```

Parameters:
- `text` (string, required): Full dialogue text
- `speaker_segments` (JSON string, required): Speaker segments with timing
- `voice1_sample` (file, optional): Voice sample for first speaker
- `voice2_sample` (file, optional): Voice sample for second speaker

### Full Video Pipeline
```bash
POST /full-video-pipeline
```

Parameters:
- `text` (string, required): Dialogue text
- `background_video` (file, optional): Background video
- `claude1_image` (file, optional): Image for first speaker
- `claude2_image` (file, optional): Image for second speaker
- `voice1_sample` (file, optional): Voice sample for first speaker
- `voice2_sample` (file, optional): Voice sample for second speaker

## Usage Examples

### Basic TTS (JSON Format)
```python
import requests

# Using Gradio API endpoint
response = requests.post(
    "https://YOUR-USERNAME-chatterbox-zero.hf.space/api/predict",
    json={
        "data": [{
            "text": "Hello, this is a test of Chatterbox TTS.",
            "voice_file": None,
            "exaggeration": 0.5,
            "cfg_weight": 0.5
        }]
    }
)

# The response contains the audio file path
result = response.json()
audio_url = result["data"][0]  # URL to generated audio file
```

### Voice Cloning (JSON Format)
```python
import requests

# Upload voice file first, then use in TTS
response = requests.post(
    "https://YOUR-USERNAME-chatterbox-zero.hf.space/api/predict",
    json={
        "data": [{
            "text": "This will sound like the voice sample.",
            "voice_file": "path/to/uploaded/voice_sample.wav",
            "exaggeration": 0.5,
            "cfg_weight": 0.5
        }]
    }
)

result = response.json()
audio_url = result["data"][0]  # URL to generated audio file
```

### Dialogue Generation (JSON Format)
```python
import requests

response = requests.post(
    "https://YOUR-USERNAME-chatterbox-zero.hf.space/api/predict",
    json={
        "data": [{
            "text": "Hello there! How are you doing today?",
            "voice1_file": None,
            "voice2_file": None,
            "speaker_segments": "[{\"speaker\":\"SPEAKER_00\",\"text\":\"Hello there!\"},{\"speaker\":\"SPEAKER_01\",\"text\":\"How are you doing today?\"}]"
        }]
    }
)

result = response.json()
audio_url = result["data"][0]  # URL to generated dialogue audio
```

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU device ID (default: "0")
- `PORT`: API port (default: 8000)

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with [Chatterbox TTS](https://github.com/jasonppy/chatterbox)
- Speaker diarization powered by [pyannote.audio](https://github.com/pyannote/pyannote-audio)