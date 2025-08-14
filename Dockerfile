FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download tiny Whisper model to reduce startup time
RUN python -c "import whisper; whisper.load_model('tiny')"

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "main.py"]