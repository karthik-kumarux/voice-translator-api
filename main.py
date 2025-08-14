from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import whisper
import tempfile
import os
import io
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI(title="Voice Assistant API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

print("Loading NLLB translation model...")
nllb_model_path = "./nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(nllb_model_path)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
translation_model.to(device)
print(f"Models loaded on device: {device}")

# Pydantic models
class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    slow: bool = False

class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "eng_Latn"  # English
    target_lang: str = "spa_Latn"  # Spanish

class FullPipelineRequest(BaseModel):
    source_lang: str = "eng_Latn"
    target_lang: str = "spa_Latn"
    tts_lang: str = "es"  # For gTTS

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    temp_file_path = None
    try:
        print(f"Received file: {audio.filename}, content_type: {audio.content_type}")
        
        # Save uploaded file temporarily
        file_extension = '.aac' if 'aac' in str(audio.filename).lower() else '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file.flush()
            temp_file_path = temp_file.name
            
        print(f"Saved temp file: {temp_file_path}")
        print(f"File size: {os.path.getsize(temp_file_path)} bytes")
        
        # Transcribe audio
        print("Starting transcription...")
        result = whisper_model.transcribe(temp_file_path)
        print(f"Transcription result: {result['text']}")
        
        return {"text": result['text']}
    
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"Cleaned up temp file: {temp_file_path}")
            except:
                pass

@app.get("/")
async def root():
    return {
        "message": "Voice Translation API",
        "version": "1.0.0",
        "endpoints": {
            "STT": "/transcribe",
            "Translation": "/translate", 
            "TTS": "/speak",
            "Full Pipeline": "/full_pipeline",
            "Health": "/health"
        },
        "supported_languages": {
            "common": {
                "English": "eng_Latn",
                "Spanish": "spa_Latn",
                "French": "fra_Latn",
                "German": "deu_Latn",
                "Italian": "ita_Latn",
                "Portuguese": "por_Latn",
                "Chinese": "zho_Hans",
                "Japanese": "jpn_Jpan",
                "Korean": "kor_Hang",
                "Arabic": "arb_Arab",
                "Hindi": "hin_Deva",
                "Russian": "rus_Cyrl"
            }
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Voice Assistant API"}

@app.post("/translate")
async def translate_text(request: TranslateRequest):
    """Translate text using NLLB model"""
    try:
        print(f"Translation request: {request.text[:50]}... ({request.source_lang} -> {request.target_lang})")
        
        # Set source language
        tokenizer.src_lang = request.source_lang
        
        # Tokenize input
        inputs = tokenizer(request.text, return_tensors="pt").to(device)
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = translation_model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[request.target_lang],
                max_length=512
            )
        
        # Decode translation
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        print(f"Translation result: {translated_text}")
        
        return {"translated_text": translated_text}
    
    except Exception as e:
        print(f"Translation Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/speak")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech and return audio file"""
    try:
        print(f"TTS request: {request.text[:50]}...")
        
        # Create TTS object
        tts = gTTS(text=request.text, lang=request.language, slow=request.slow)
        
        # Save to memory buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        print(f"Generated TTS audio: {len(audio_buffer.getvalue())} bytes")
        
        return StreamingResponse(
            io.BytesIO(audio_buffer.getvalue()),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

@app.post("/full_pipeline")
async def full_translation_pipeline(audio: UploadFile = File(...), source_lang: str = "eng_Latn", target_lang: str = "spa_Latn", tts_lang: str = "es"):
    """Complete pipeline: Audio -> STT -> Translation -> TTS"""
    temp_file_path = None
    try:
        print(f"Full pipeline: {source_lang} -> {target_lang}")
        
        # Step 1: STT (Speech to Text)
        file_extension = '.aac' if 'aac' in str(audio.filename).lower() else '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file.flush()
            temp_file_path = temp_file.name
        
        print("Step 1: Transcribing audio...")
        result = whisper_model.transcribe(temp_file_path)
        original_text = result['text']
        print(f"Transcribed: {original_text}")
        
        # Step 2: Translation
        print("Step 2: Translating text...")
        tokenizer.src_lang = source_lang
        inputs = tokenizer(original_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_tokens = translation_model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
                max_length=512
            )
        
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        print(f"Translated: {translated_text}")
        
        # Step 3: TTS (Text to Speech)
        print("Step 3: Generating speech...")
        tts = gTTS(text=translated_text, lang=tts_lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return {
            "original_text": original_text,
            "translated_text": translated_text,
            "audio_url": "/get_audio"  # We'll return the audio separately
        }
    
    except Exception as e:
        print(f"Full pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)