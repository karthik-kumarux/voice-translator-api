# Voice Translation API

A complete voice translation service with Speech-to-Text, Translation, and Text-to-Speech capabilities.

## Features

- **STT**: Whisper model for speech recognition
- **Translation**: NLLB-200 model for multilingual translation  
- **TTS**: Google Text-to-Speech for audio output
- **Full Pipeline**: Complete voice-to-voice translation

## API Endpoints

- `POST /transcribe` - Audio to text
- `POST /translate` - Text translation
- `POST /speak` - Text to speech
- `POST /full_pipeline` - Complete voice translation
- `GET /health` - Health check

## Supported Languages

English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, Hindi, Russian

## Deployment

This service is deployed on Railway and ready for production use.