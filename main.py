from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import tempfile
import os
import io
from gtts import gTTS
import requests
import openai

app = FastAPI(title="Voice Translation API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API configurations
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

openai.api_key = os.environ.get("OPENAI_API_KEY", "")
print("Voice Translation API ready")

# Pydantic models
class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "eng_Latn"
    target_lang: str = "spa_Latn"

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

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
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Voice Translation API"}

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    temp_file_path = None
    try:
        if not openai.api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file.flush()
            temp_file_path = temp_file.name
            
        with open(temp_file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        
        return {"text": transcript.text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass

@app.post("/translate")
async def translate_text(request: TranslateRequest):
    try:
        if not HF_TOKEN:
            raise HTTPException(status_code=500, detail="HF_TOKEN not set")
            
        payload = {
            "inputs": request.text,
            "parameters": {
                "src_lang": request.source_lang,
                "tgt_lang": request.target_lang
            }
        }
        
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            translated_text = result[0].get('translation_text', request.text)
        else:
            translated_text = request.text
        
        return {"translated_text": translated_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/speak")
async def text_to_speech(request: TTSRequest):
    try:
        tts = gTTS(text=request.text, lang=request.language, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(audio_buffer.getvalue()),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

@app.post("/full_pipeline")
async def full_translation_pipeline(audio: UploadFile = File(...), source_lang: str = "eng_Latn", target_lang: str = "spa_Latn", tts_lang: str = "es"):
    temp_file_path = None
    try:
        if not openai.api_key or not HF_TOKEN:
            raise HTTPException(status_code=500, detail="API keys not configured")
            
        # Step 1: STT
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file.flush()
            temp_file_path = temp_file.name
        
        with open(temp_file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        original_text = transcript.text
        
        # Step 2: Translation
        payload = {
            "inputs": original_text,
            "parameters": {
                "src_lang": source_lang,
                "tgt_lang": target_lang
            }
        }
        
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            translated_text = result[0].get('translation_text', original_text)
        else:
            translated_text = original_text
        
        return {
            "original_text": original_text,
            "translated_text": translated_text
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)