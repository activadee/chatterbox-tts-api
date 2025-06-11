# app.py für dein Hugging Face Space mit Zero GPU
import gradio as gr
import torch
import torchaudio as ta
import spaces
import tempfile
import os
import requests
import urllib.parse
import re
import json
from pathlib import Path
from chatterbox.tts import ChatterboxTTS

# Model wird lazy geladen
model = None

def get_model():
    """Lazy loading des Models"""
    global model
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxTTS.from_pretrained(device=device)
    return model

def resolve_google_drive_url(url):
    """Convert Google Drive sharing URL to direct download URL"""
    if "drive.google.com" not in url:
        return url
    
    # Extract file ID from various Google Drive URL formats
    patterns = [
        r"/file/d/([a-zA-Z0-9-_]+)",
        r"id=([a-zA-Z0-9-_]+)",
        r"/d/([a-zA-Z0-9-_]+)"
    ]
    
    file_id = None
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            file_id = match.group(1)
            break
    
    if file_id:
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    
    return url

def download_audio_file(file_input):
    """
    Handle file input - can be local file, URL, or Google Drive link
    Returns path to local file
    """
    if file_input is None:
        return None
    
    # If it's already a local file path (from Gradio file upload)
    if hasattr(file_input, 'name'):
        return file_input.name
    
    # If it's a string, check if it's a URL or local path
    if isinstance(file_input, str):
        # Check if it's a URL
        if file_input.startswith(('http://', 'https://')):
            try:
                # Resolve Google Drive URLs
                download_url = resolve_google_drive_url(file_input)
                
                # Download the file
                response = requests.get(download_url, stream=True, timeout=30, allow_redirects=True)
                response.raise_for_status()
                
                # Get file extension from Content-Type or URL
                content_type = response.headers.get('content-type', '')
                if 'audio' in content_type:
                    if 'wav' in content_type:
                        ext = '.wav'
                    elif 'mp3' in content_type:
                        ext = '.mp3'
                    elif 'ogg' in content_type:
                        ext = '.ogg'
                    else:
                        ext = '.wav'  # default
                else:
                    # Try to get extension from URL
                    parsed_url = urllib.parse.urlparse(download_url)
                    path = Path(parsed_url.path)
                    ext = path.suffix if path.suffix in ['.wav', '.mp3', '.ogg', '.m4a', '.flac'] else '.wav'
                
                # Save to temporary file
                temp_file = tempfile.mktemp(suffix=ext)
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"Downloaded audio file from {file_input} to {temp_file}")
                return temp_file
                
            except Exception as e:
                print(f"Failed to download audio from {file_input}: {e}")
                return None
        
        # If it's a local file path
        elif os.path.exists(file_input):
            return file_input
    
    return file_input

@spaces.GPU  # Zero GPU Decorator - wichtig!
def generate_speech(text, voice_file, voice_url, exaggeration, cfg_weight):
    """TTS Generation mit Zero GPU - direkte Argumente"""
    try:
        if not text.strip():
            return None, "Please enter some text"
        
        model = get_model()
        
        # Prepare kwargs
        kwargs = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight
        }
        
        # Handle voice input - prioritize URL over file upload
        voice_input = voice_url.strip() if voice_url and voice_url.strip() else voice_file
        
        if voice_input:
            audio_path = download_audio_file(voice_input)
            if audio_path:
                kwargs["audio_prompt_path"] = audio_path
                print(f"Using voice sample: {audio_path}")
            else:
                return None, "Failed to download or process voice file"
        
        # Generate audio
        wav = model.generate(text, **kwargs)
        
        # Save to temporary file
        output_path = tempfile.mktemp(suffix=".wav")
        ta.save(output_path, wav, model.sr)
        
        print(f"Generated speech saved to {output_path}")
        return output_path, "✅ Speech generated successfully"
 
    except Exception as e:
        print(f"Error in generate_speech: {e}")
        return None, f"❌ Error: {str(e)}"

@spaces.GPU
def generate_dialogue(text, voice1_file, voice1_url, voice2_file, voice2_url, speaker_segments_str):
    """Multi-speaker dialogue generation - directe Argumente"""
    import json
    try:
        if not text.strip():
            return None, "Please enter some text"
        
        segments = json.loads(speaker_segments_str) if speaker_segments_str else []
        
        if not segments:
            # Simple split for demo
            words = text.split()
            mid = len(words) // 2
            segments = [
                {"speaker": "SPEAKER_00", "text": " ".join(words[:mid])},
                {"speaker": "SPEAKER_01", "text": " ".join(words[mid:])}
            ]
        
        model = get_model()
        audio_segments = []
        
        # Handle voice files - prioritize URLs over file uploads
        voice_files = {}
        
        # Speaker 1 voice
        voice1_input = voice1_url.strip() if voice1_url and voice1_url.strip() else voice1_file
        if voice1_input:
            voice1_path = download_audio_file(voice1_input)
            if voice1_path:
                voice_files["SPEAKER_00"] = voice1_path
                print(f"Using Speaker 1 voice: {voice1_path}")
            else:
                return None, "Failed to download or process voice file for Speaker 1"
        
        # Speaker 2 voice
        voice2_input = voice2_url.strip() if voice2_url and voice2_url.strip() else voice2_file
        if voice2_input:
            voice2_path = download_audio_file(voice2_input)
            if voice2_path:
                voice_files["SPEAKER_01"] = voice2_path
                print(f"Using Speaker 2 voice: {voice2_path}")
            else:
                return None, "Failed to download or process voice file for Speaker 2"
        
        for segment in segments:
            speaker = segment["speaker"]
            segment_text = segment["text"]
            
            kwargs = {"exaggeration": 0.5, "cfg_weight": 0.5}
            if voice_files.get(speaker):
                kwargs["audio_prompt_path"] = voice_files[speaker]
            
            wav = model.generate(segment_text, **kwargs)
            audio_segments.append(wav)
        
        # Concatenate segments
        if audio_segments:
            full_audio = torch.cat(audio_segments, dim=-1)
        else:
            full_audio = model.generate(text)
        
        # Save result
        output_path = tempfile.mktemp(suffix=".wav")
        ta.save(output_path, full_audio, model.sr)
        
        return output_path, f"✅ Generated dialogue with {len(segments)} speakers"
        
    except json.JSONDecodeError as e: # Specific to speaker_segments_str parsing
        return None, f"❌ Invalid Speaker Segments JSON: {str(e)}"
    except Exception as e:
        print(f"Error in generate_dialogue: {e}")
        return None, f"❌ Error: {str(e)}"

# Gradio Interface
with gr.Blocks(title="Chatterbox TTS API", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 Chatterbox TTS with Zero GPU")
    gr.Markdown("High-quality text-to-speech with voice cloning and emotion control")
    
    with gr.Tab("🗣️ Basic TTS"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text to speak",
                    placeholder="Enter your text here...",
                    lines=3,
                    value="Hello! This is Chatterbox TTS running on Zero GPU."
                )
                voice_file = gr.File(
                    label="Voice Sample (Upload File - optional)",
                    file_types=["audio"],
                    type="filepath"
                )
                
                voice_url = gr.Textbox(
                    label="Voice Sample URL (Google Drive or direct link - optional)",
                    placeholder="https://drive.google.com/file/d/... or direct audio URL",
                    info="Leave empty if using file upload above. Supports Google Drive sharing links."
                )
                
                with gr.Row():
                    exaggeration = gr.Slider(
                        0, 1, value=0.5, step=0.1,
                        label="Exaggeration",
                        info="Higher = more dramatic"
                    )
                    cfg_weight = gr.Slider(
                        0, 1, value=0.5, step=0.1,
                        label="CFG Weight", 
                        info="Lower = slower pace"
                    )
                
                generate_btn = gr.Button("🎵 Generate Speech", variant="primary")
            
            with gr.Column():
                audio_output = gr.Audio(label="Generated Speech")
                status_output = gr.Textbox(label="Status", interactive=False)
    
    with gr.Tab("🎭 Dialogue Generation"):
        with gr.Row():
            with gr.Column():
                dialogue_text = gr.Textbox(
                    label="Dialogue Text",
                    placeholder="Enter dialogue or text to split between speakers...",
                    lines=4
                )
                
                voice1_file = gr.File(
                    label="Speaker 1 Voice Sample (Upload File - optional)",
                    file_types=["audio"],
                    type="filepath"
                )
                voice1_url = gr.Textbox(
                    label="Speaker 1 Voice URL (Google Drive or direct link - optional)",
                    placeholder="https://drive.google.com/file/d/... or direct audio URL",
                    info="Leave empty if using file upload above."
                )
                
                voice2_file = gr.File(
                    label="Speaker 2 Voice Sample (Upload File - optional)", 
                    file_types=["audio"],
                    type="filepath"
                )
                voice2_url = gr.Textbox(
                    label="Speaker 2 Voice URL (Google Drive or direct link - optional)",
                    placeholder="https://drive.google.com/file/d/... or direct audio URL",
                    info="Leave empty if using file upload above."
                )
                
                speaker_segments = gr.Textbox(
                    label="Speaker Segments (JSON, optional)",
                    placeholder='[{"speaker": "SPEAKER_00", "text": "Hello"}, {"speaker": "SPEAKER_01", "text": "Hi there"}]',
                    lines=3
                )
                
                dialogue_btn = gr.Button("🎭 Generate Dialogue", variant="primary")
            
            with gr.Column():
                dialogue_audio = gr.Audio(label="Generated Dialogue")
                dialogue_status = gr.Textbox(label="Status", interactive=False)
    
    with gr.Tab("📖 API Usage"):
        gr.Markdown("""
        ## API Endpoints

        The API expects arguments to be passed as a list in the `data` field for HTTP requests, or directly as arguments for `gradio_client`.
        
        ### Basic TTS
        Corresponds to the first endpoint (`api_name="/predict"`).
        Function signature: `generate_speech(text, voice_file, voice_url, exaggeration, cfg_weight)`

        **`gradio_client` Example:**
        ```python
        from gradio_client import Client

        client = Client("YOUR-USERNAME/chatterbox-zero") # Replace with your Space ID
        result = client.predict(
            "Hello world!",    # text (string)
            None,              # voice_file (None or path to uploaded audio file)
            "https://drive.google.com/file/d/1ABC123/view", # voice_url (Google Drive or direct URL)
            0.5,               # exaggeration (float)
            0.5,               # cfg_weight (float)
            api_name="/predict"
        )
        print(f"Generated audio saved at: {result[0]}") # result is a tuple (output_audio_path, status_message)
        # or print(f"Error: {result[1]}") if an error occurred
        ```
        
        **n8n / HTTP Request Example:**
        ```json
        POST https://YOUR-USERNAME-chatterbox-zero.hf.space/api/predict
        {
          "data": [
            "Your text here", // text (string)
            null,             // voice_file (None or local file path)
            "https://drive.google.com/file/d/1ABC123/view", // voice_url (Google Drive or direct URL)
            0.5,              // exaggeration (float)
            0.5               // cfg_weight (float)
          ]
        }
        ```
        
        ### Dialogue Generation
        Corresponds to the second endpoint (`api_name="/predict_1"`).
        Function signature: `generate_dialogue(text, voice1_file, voice1_url, voice2_file, voice2_url, speaker_segments_str)`

        **`gradio_client` Example:**
        ```python
        from gradio_client import Client

        client = Client("YOUR-USERNAME/chatterbox-zero") # Replace with your Space ID
        result = client.predict(
            "Speaker 1: Hello! Speaker 2: Hi there.", # text (string, full dialogue text)
            None,                                     # voice1_file (None or path to Speaker 1 audio)
            "https://drive.google.com/file/d/1ABC123/view", # voice1_url (Google Drive or direct URL)
            None,                                     # voice2_file (None or path to Speaker 2 audio)
            "https://drive.google.com/file/d/1DEF456/view", # voice2_url (Google Drive or direct URL)
            "[{\"speaker\":\"SPEAKER_00\",\"text\":\"Hello!\"}, {\"speaker\":\"SPEAKER_01\",\"text\":\"Hi there.\"}]", # speaker_segments_str (JSON string)
            api_name="/predict_1"
        )
        print(f"Generated dialogue audio saved at: {result[0]}") # result is a tuple (output_audio_path, status_message)
        # or print(f"Error: {result[1]}") if an error occurred
        ```

        **n8n / HTTP Request Example:**
        ```json
        POST https://YOUR-USERNAME-chatterbox-zero.hf.space/api/predict_1
        {
          "data": [
            "Speaker 1: Hello! Speaker 2: Hi there.", // text (string, full dialogue text)
            null,                                     // voice1_file (None or local file path)
            "https://drive.google.com/file/d/1ABC123/view", // voice1_url (Google Drive or direct URL)
            null,                                     // voice2_file (None or local file path)
            "https://drive.google.com/file/d/1DEF456/view", // voice2_url (Google Drive or direct URL)
            "[{\"speaker\":\"SPEAKER_00\",\"text\":\"Hello!\"}, {\"speaker\":\"SPEAKER_01\",\"text\":\"Hi there.\"}]" // speaker_segments_str (JSON string)
          ]
        }
        ```
        
        ### Voice Cloning:
        - **File Upload**: Use the file upload interface in the web UI
        - **URLs**: Pass Google Drive sharing links or direct audio URLs
        - **Google Drive**: Sharing links like `https://drive.google.com/file/d/1ABC123/view` are automatically converted to direct download links
        - **Supported formats**: WAV, MP3, OGG, M4A, FLAC
        """)
    
    # Event handlers
    generate_btn.click(
        generate_speech,
        inputs=[text_input, voice_file, voice_url, exaggeration, cfg_weight],
        outputs=[audio_output, status_output]
    )
    
    dialogue_btn.click(
        generate_dialogue,
        inputs=[dialogue_text, voice1_file, voice1_url, voice2_file, voice2_url, speaker_segments],
        outputs=[dialogue_audio, dialogue_status]
    )

# Launch
if __name__ == "__main__":
    demo.launch(show_api=True)