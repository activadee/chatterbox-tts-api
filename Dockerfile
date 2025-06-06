# Dockerfile for Standalone Chatterbox TTS API
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    chatterbox-tts==0.1.1 \
    fastapi[standard] \
    uvicorn \
    python-multipart \
    pyannote.audio \
    speechbrain \
    torch \
    torchaudio \
    numpy

# Copy application files
COPY app/ .

# Create directories for uploads and outputs
RUN mkdir -p /app/uploads /app/outputs /app/voices

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]