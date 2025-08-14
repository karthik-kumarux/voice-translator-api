from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests

app = FastAPI(title="Translation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HuggingFace API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

print("Translation API ready")

class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "eng_Latn"
    target_lang: str = "spa_Latn"

@app.get("/")
async def root():
    return {
        "message": "Translation API",
        "version": "1.0.0",
        "status": "working"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Translation API"}

@app.post("/translate")
async def translate_text(request: TranslateRequest):
    try:
        if not HF_TOKEN:
            return {"translated_text": f"[Translation] {request.text}"}
            
        payload = {
            "inputs": request.text,
            "parameters": {
                "src_lang": request.source_lang,
                "tgt_lang": request.target_lang
            }
        }
        
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                translated_text = result[0].get('translation_text', request.text)
            else:
                translated_text = request.text
        else:
            translated_text = f"[Mock Translation] {request.text}"
        
        return {"translated_text": translated_text}
    
    except Exception as e:
        return {"translated_text": f"[Error] {request.text}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)