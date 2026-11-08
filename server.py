#!/usr/bin/env python3
"""faster-whisper STT service — large-v3-turbo, int8_float16 on GPU0. Port 18080."""
import io, threading, time
from flask import Flask, jsonify, request

MODEL = "h2oai/faster-whisper-large-v3-turbo"
app = Flask(__name__)
_lock = threading.Lock()
_model = None
_loaded_at = None


def get_model():
    global _model, _loaded_at
    if _model is None:
        from faster_whisper import WhisperModel
        t0 = time.time()
        _model = WhisperModel(MODEL, device="cuda", compute_type="int8_float16")
        _loaded_at = round(time.time() - t0, 1)
        print(f"[whisper] model ready in {_loaded_at}s", flush=True)
    return _model


@app.get("/health")
def health():
    return jsonify(status="ok", model_loaded=_model is not None, load_seconds=_loaded_at)


@app.post("/transcribe")
def transcribe():
    f = request.files.get("file")
    if f is None:
        return jsonify(error="multipart field 'file' required (wav/mp3/m4a/ogg/flac...)"), 400
    language = request.args.get("language") or None  # "ru"/"en"/... or None=auto
    try:
        beam = int(request.args.get("beam", 5))
    except ValueError:
        beam = 5
    with _lock:
        model = get_model()
        t0 = time.time()
        segments, info = model.transcribe(
            io.BytesIO(f.read()),
            language=language,
            beam_size=beam,
            vad_filter=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
    return jsonify(
        text=text,
        language=info.language,
        probability=round(info.language_probability, 3),
        audio_duration=round(info.duration, 2),
        elapsed=round(time.time() - t0, 2),
    )


if __name__ == "__main__":
    get_model()
    app.run(host="0.0.0.0", port=18080, threaded=True)
