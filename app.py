# app.py für dein Hugging Face Space mit Zero GPU
import gradio as gr
import torch
import torchaudio as ta
import spaces
import tempfile
import os
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

@spaces.GPU  # Zero GPU Decorator - wichtig!
def generate_speech(text, voice_file, exaggeration, cfg_weight):
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
        
        # Voice cloning wenn file hochgeladen
        if voice_file is not None:
            kwargs["audio_prompt_path"] = voice_file.name if hasattr(voice_file, 'name') else voice_file
        
        # Generate audio
        wav = model.generate(text, **kwargs)
        
        # Save to temporary file
        output_path = tempfile.mktemp(suffix=".wav")
        ta.save(output_path, wav, model.sr)
        # logging
        print(f"Generated speech saved to {output_path}")
        return output_path, None
 
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

@spaces.GPU
def generate_dialogue(text, voice1_file, voice2_file, speaker_segments_str):
    """Multi-speaker dialogue generation - directe Argumente"""
    try:
        import json
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
        
        # Voice mapping
        voice_files = {
            "SPEAKER_00": voice1_file.name if (voice1_file and hasattr(voice1_file, 'name')) else voice1_file,
            "SPEAKER_01": voice2_file.name if (voice2_file and hasattr(voice2_file, 'name')) else voice2_file
        }
        
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
                    label="Voice Sample (optional)",
                    file_types=["audio"],
                    type="filepath"
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
                    label="Speaker 1 Voice Sample",
                    file_types=["audio"],
                    type="filepath"
                )
                voice2_file = gr.File(
                    label="Speaker 2 Voice Sample", 
                    file_types=["audio"],
                    type="filepath"
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
        Function signature: `generate_speech(text, voice_file, exaggeration, cfg_weight)`

        **`gradio_client` Example:**
        ```python
        from gradio_client import Client

        client = Client("YOUR-USERNAME/chatterbox-zero") # Replace with your Space ID
        result = client.predict(
            "Hello world!",    # text (string)
            None,              # voice_file (None or path to uploaded audio file)
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
            null,             // voice_file (None or URL/path to audio if server setup allows remote fetching)
            0.5,              // exaggeration (float)
            0.5               // cfg_weight (float)
          ]
        }
        ```
        
        ### Dialogue Generation
        Corresponds to the second endpoint (`api_name="/predict_1"`).
        Function signature: `generate_dialogue(text, voice1_file, voice2_file, speaker_segments_str)`

        **`gradio_client` Example:**
        ```python
        from gradio_client import Client

        client = Client("YOUR-USERNAME/chatterbox-zero") # Replace with your Space ID
        result = client.predict(
            "Speaker 1: Hello! Speaker 2: Hi there.", # text (string, full dialogue text)
            None,                                     # voice1_file (None or path to Speaker 1 audio)
            None,                                     # voice2_file (None or path to Speaker 2 audio)
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
            null,                                     // voice1_file (None or URL/path)
            null,                                     // voice2_file (None or URL/path)
            "[{\"speaker\":\"SPEAKER_00\",\"text\":\"Hello!\"}, {\"speaker\":\"SPEAKER_01\",\"text\":\"Hi there.\"}]" // speaker_segments_str (JSON string)
          ]
        }
        ```
        
        ### Voice Cloning:
        Upload audio file in the interface or pass file path via API.
        """)
    
    # Event handlers
    generate_btn.click(
        generate_speech,
        inputs=[text_input, voice_file, exaggeration, cfg_weight],
        outputs=[audio_output, status_output]
    )
    
    dialogue_btn.click(
        generate_dialogue,
        inputs=[dialogue_text, voice1_file, voice2_file, speaker_segments],
        outputs=[dialogue_audio, dialogue_status]
    )

# Launch
if __name__ == "__main__":
    demo.launch(show_api=True)