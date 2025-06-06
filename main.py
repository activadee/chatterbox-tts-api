# app/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
import torchaudio as ta
import io
import os
import json
import uuid
import tempfile
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chatterbox TTS API",
    description="Standalone Chatterbox TTS service with voice cloning and speaker diarization",
    version="1.0.0"
)

# Global model instance
model = None

class TTSRequest(BaseModel):
    text: str
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    voice_name: Optional[str] = None
    return_format: str = "wav"  # wav, mp3
    split_long_text: bool = True
    chunk_size: int = 1000

class SpeakerDiarizationRequest(BaseModel):
    return_timestamps: bool = True
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

class VideoGenerationRequest(BaseModel):
    text: str
    background_video_url: Optional[str] = None
    claude1_image_url: Optional[str] = None
    claude2_image_url: Optional[str] = None
    voice1_sample: Optional[str] = None
    voice2_sample: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Load the Chatterbox model on startup"""
    global model
    try:
        logger.info("Loading Chatterbox TTS model...")
        from chatterbox.tts import ChatterboxTTS
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("Chatterbox TTS model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    return_format: str = Form("wav"),
    voice_sample: Optional[UploadFile] = File(None)
):
    """Generate speech from text with optional voice cloning"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Handle voice cloning if sample provided
        audio_prompt_path = None
        if voice_sample:
            # Save uploaded voice sample temporarily
            temp_dir = tempfile.gettempdir()
            voice_path = os.path.join(temp_dir, f"voice_{uuid.uuid4().hex}.wav")
            
            with open(voice_path, "wb") as f:
                content = await voice_sample.read()
                f.write(content)
            audio_prompt_path = voice_path
        
        # Generate speech
        kwargs = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight
        }
        
        if audio_prompt_path:
            kwargs["audio_prompt_path"] = audio_prompt_path
        
        wav = model.generate(text, **kwargs)
        
        # Clean up temporary voice file
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            os.remove(audio_prompt_path)
        
        # Convert to audio file
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        
        # Return audio file
        media_type = f"audio/{return_format}"
        filename = f"chatterbox_{uuid.uuid4().hex}.{return_format}"
        
        return StreamingResponse(
            buffer,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speaker-diarization")
async def speaker_diarization(
    audio_file: UploadFile = File(...),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None)
):
    """Perform speaker diarization on audio file"""
    try:
        # Save uploaded audio temporarily
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"audio_{uuid.uuid4().hex}.wav")
        
        with open(audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Perform speaker diarization with pyannote
        from pyannote.audio import Pipeline
        
        # Load pre-trained pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        
        # Apply pipeline
        diarization = pipeline(audio_path)
        
        # Convert to segments list
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker,
                "duration": float(turn.end - turn.start)
            })
        
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return {
            "segments": segments,
            "total_speakers": len(set(seg["speaker"] for seg in segments)),
            "total_duration": max(seg["end"] for seg in segments) if segments else 0
        }
        
    except Exception as e:
        logger.error(f"Speaker diarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-dialogue-audio")
async def generate_dialogue_audio(
    text: str = Form(...),
    speaker_segments: str = Form(...),  # JSON string
    voice1_sample: Optional[UploadFile] = File(None),
    voice2_sample: Optional[UploadFile] = File(None)
):
    """Generate dialogue audio with different voices for different speakers"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Parse speaker segments
        segments = json.loads(speaker_segments)
        
        # Save voice samples temporarily
        voice_paths = {}
        if voice1_sample:
            voice1_path = os.path.join(tempfile.gettempdir(), f"voice1_{uuid.uuid4().hex}.wav")
            with open(voice1_path, "wb") as f:
                content = await voice1_sample.read()
                f.write(content)
            voice_paths["SPEAKER_00"] = voice1_path
        
        if voice2_sample:
            voice2_path = os.path.join(tempfile.gettempdir(), f"voice2_{uuid.uuid4().hex}.wav")
            with open(voice2_path, "wb") as f:
                content = await voice2_sample.read()
                f.write(content)
            voice_paths["SPEAKER_01"] = voice2_path
        
        # Generate audio for each segment
        audio_segments = []
        sample_rate = model.sr
        
        for segment in segments:
            speaker = segment["speaker"]
            segment_text = segment.get("text", "")
            
            if not segment_text.strip():
                continue
            
            # Use appropriate voice
            kwargs = {"exaggeration": 0.5, "cfg_weight": 0.5}
            if speaker in voice_paths:
                kwargs["audio_prompt_path"] = voice_paths[speaker]
            
            # Generate audio for this segment
            wav = model.generate(segment_text, **kwargs)
            audio_segments.append(wav)
        
        # Concatenate all segments
        if audio_segments:
            full_audio = torch.cat(audio_segments, dim=-1)
        else:
            # Generate with default voice if no segments
            full_audio = model.generate(text)
        
        # Clean up voice samples
        for path in voice_paths.values():
            if os.path.exists(path):
                os.remove(path)
        
        # Return audio
        buffer = io.BytesIO()
        ta.save(buffer, full_audio, sample_rate, format="wav")
        buffer.seek(0)
        
        filename = f"dialogue_{uuid.uuid4().hex}.wav"
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Dialogue generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/full-video-pipeline")
async def full_video_pipeline(
    text: str = Form(...),
    background_video: Optional[UploadFile] = File(None),
    claude1_image: Optional[UploadFile] = File(None),
    claude2_image: Optional[UploadFile] = File(None),
    voice1_sample: Optional[UploadFile] = File(None),
    voice2_sample: Optional[UploadFile] = File(None)
):
    """Complete pipeline: TTS → Speaker Diarization → Video Generation"""
    try:
        # Step 1: Generate base audio
        logger.info("Step 1: Generating base audio...")
        
        # Generate TTS for the full text first
        wav = model.generate(text, exaggeration=0.5, cfg_weight=0.5)
        
        # Save temporary audio file for diarization
        temp_audio_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{uuid.uuid4().hex}.wav")
        ta.save(temp_audio_path, wav, model.sr)
        
        # Step 2: Perform speaker diarization (simulated - you'd need actual dialogue)
        logger.info("Step 2: Performing speaker diarization...")
        
        # For now, split text roughly in half for two speakers
        words = text.split()
        mid_point = len(words) // 2
        
        segments = [
            {
                "start": 0.0,
                "end": len(words[:mid_point]) * 0.1,  # Rough timing
                "speaker": "SPEAKER_00",
                "text": " ".join(words[:mid_point])
            },
            {
                "start": len(words[:mid_point]) * 0.1,
                "end": len(words) * 0.1,
                "speaker": "SPEAKER_01", 
                "text": " ".join(words[mid_point:])
            }
        ]
        
        # Step 3: Generate dialogue audio with different voices
        logger.info("Step 3: Generating dialogue audio...")
        
        # Save voice samples
        voice_paths = {}
        if voice1_sample:
            voice1_path = os.path.join(tempfile.gettempdir(), f"voice1_{uuid.uuid4().hex}.wav")
            with open(voice1_path, "wb") as f:
                content = await voice1_sample.read()
                f.write(content)
            voice_paths["SPEAKER_00"] = voice1_path
        
        if voice2_sample:
            voice2_path = os.path.join(tempfile.gettempdir(), f"voice2_{uuid.uuid4().hex}.wav")
            with open(voice2_path, "wb") as f:
                content = await voice2_sample.read()
                f.write(content)
            voice_paths["SPEAKER_01"] = voice2_path
        
        # Generate final audio
        audio_segments = []
        for segment in segments:
            speaker = segment["speaker"]
            segment_text = segment["text"]
            
            kwargs = {"exaggeration": 0.5, "cfg_weight": 0.5}
            if speaker in voice_paths:
                kwargs["audio_prompt_path"] = voice_paths[speaker]
            
            wav_segment = model.generate(segment_text, **kwargs)
            audio_segments.append(wav_segment)
        
        # Concatenate segments
        final_audio = torch.cat(audio_segments, dim=-1)
        
        # Save final audio
        final_audio_path = os.path.join(tempfile.gettempdir(), f"final_audio_{uuid.uuid4().hex}.wav")
        ta.save(final_audio_path, final_audio, model.sr)
        
        # Step 4: Prepare video data (for now just return info)
        result = {
            "status": "success",
            "segments": segments,
            "audio_duration": float(final_audio.shape[-1]) / model.sr,
            "message": "Audio generated successfully. Use FFmpeg separately for video generation.",
            "ffmpeg_command_template": """
            ffmpeg -i {background_video} -i {final_audio} -i {claude1_image} -i {claude2_image} \\
            -filter_complex "[0:v][2:v]overlay=W-w-10:10:enable='between(t,{start1},{end1})'[v1];[v1][3:v]overlay=W-w-10:10:enable='between(t,{start2},{end2})'[v2]" \\
            -map "[v2]" -map 1:a -c:v libx264 -c:a aac output.mp4
            """.format(
                background_video="background.mp4",
                final_audio="final_audio.wav", 
                claude1_image="claude1.png",
                claude2_image="claude2.png",
                start1=segments[0]["start"],
                end1=segments[0]["end"],
                start2=segments[1]["start"],
                end2=segments[1]["end"]
            )
        }
        
        # Return final audio
        buffer = io.BytesIO()
        ta.save(buffer, final_audio, model.sr, format="wav")
        buffer.seek(0)
        
        # Clean up
        for path in [temp_audio_path, final_audio_path] + list(voice_paths.values()):
            if os.path.exists(path):
                os.remove(path)
        
        filename = f"pipeline_result_{uuid.uuid4().hex}.wav"
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Pipeline-Info": json.dumps(result)
            }
        )
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)