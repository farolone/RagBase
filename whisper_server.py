"""Whisper Transcription Server for Mac Studio Ultra.

Provides an OpenAI-compatible /v1/audio/transcriptions endpoint
using mlx-whisper for fast Apple Silicon inference.

Install:
    pip install mlx-whisper fastapi uvicorn python-multipart

Run:
    python whisper_server.py

Listens on port 8765. Test:
    curl -X POST http://localhost:8765/v1/audio/transcriptions \
         -F file=@audio.mp3 -F model=mlx-community/whisper-large-v3-turbo
"""

import tempfile
from pathlib import Path

import mlx_whisper
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="Whisper Server")

# Pre-load model on startup
MODEL = "mlx-community/whisper-large-v3-turbo"
print(f"Loading model {MODEL}...")
# Warm up with a dummy transcription to download and load the model
try:
    mlx_whisper.transcribe("/dev/null", path_or_hf_repo=MODEL)
except Exception:
    pass  # Expected to fail on /dev/null, but model is now cached
print("Model ready!")


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=MODEL),
    language: str = Form(default=None),
    response_format: str = Form(default="verbose_json"),
):
    """OpenAI-compatible transcription endpoint."""
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = mlx_whisper.transcribe(
            tmp_path,
            path_or_hf_repo=model or MODEL,
            language=language,
            word_timestamps=True,
        )

        if response_format == "text":
            return JSONResponse({"text": result["text"]})

        # verbose_json format (compatible with OpenAI)
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "id": seg.get("id", 0),
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            })

        return JSONResponse({
            "text": result["text"],
            "language": result.get("language", ""),
            "segments": segments,
        })
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
