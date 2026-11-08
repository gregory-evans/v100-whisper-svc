# v100-whisper-svc

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) speech-to-text as a tiny HTTP service for an NVIDIA GPU node. Runs **on the same GPU next to the LLM** (measured footprint: **+1.3 GB VRAM**), transcribes at **~10× realtime**.

Client side lives in [`pi-v100-media`](https://github.com/gregory-evans/pi-v100-media) (Pi coding agent extension).

## Files

| File | Purpose |
|---|---|
| `server.py` | Flask service, 60 lines: lazy model load, `/health`, `/transcribe` |
| `requirements.txt` | **Pinned** dependencies — do not "upgrade" casually (see Gotchas) |
| `whisper.service` | systemd unit (paths for a Debian/Ubuntu server, Python 3.13 venv at `/root/whisper-svc`) |

## Requirements

- Linux x86_64 with an NVIDIA CUDA GPU. Tested: **Tesla V100 32 GB**, driver 580.x, Debian 13.
  - Any modern GPU works; the V100 (sm_70) is fine because `int8_float16` compute doesn't need int8 tensor cores.
- Python **3.10–3.13** with `venv`. Tested: 3.13.
- Disk: ~400 MB for the venv (cudnn/cublas wheels) + **~1.6 GB model cache** on first run (`~/.cache/huggingface`).
- VRAM in steady state: **≈ 1.3–2 GB** (weights `int8_float16` ~1.5 GB + a second CUDA context).

## Deployment

```bash
# 1) Files to /root/whisper-svc  (any directory works — adjust the unit accordingly)
sudo mkdir -p /root/whisper-svc
cp server.py requirements.txt /root/whisper-svc/
cd /root/whisper-svc

# 2) venv + pinned deps (~400 MB; includes nvidia-cudnn-cu12 / nvidia-cublas-cu12 wheels — no system CUDA libs needed)
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# 3) systemd unit (check WorkingDirectory + LD_LIBRARY_PATH if your paths differ!)
sudo cp whisper.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now whisper

# 4) Health — FIRST START downloads the model (~1.6 GB), can take up to ~15 min:
curl http://localhost:18080/health
# {"load_seconds":69.4,"model_loaded":true,"status":"ok"}     ← first boot (cold download+load)
# {"load_seconds":7.2,"model_loaded":true,"status":"ok"}      ← subsequent restarts (cached)

# 5) Smoke test:
curl -F "file=@/path/to/audio.wav" http://localhost:18080/transcribe
```

## API

| Endpoint | Description |
|---|---|
| `GET /health` | `{status, model_loaded, load_seconds}` — model is loaded lazily on first request/startup |
| `POST /transcribe` | Multipart field **`file`** (wav/mp3/m4a/ogg/opus/flac/webm/video with audio track…). Optional query params: `language=ru\|en\|…` (default: auto-detect), `beam=N` (default 5) |

Response:

```json
{"text": "And so, my fellow Americans...", "language": "en", "probability": 0.958,
 "audio_duration": 11.0, "elapsed": 1.08}
```

Notes: VAD filter is on (silence skipped automatically); a global lock serializes requests (fine for voice-note workloads — one at a time).

## VRAM budget on the shared GPU (measured)

| Component | VRAM |
|---|---|
| Qwen3.8-27B Q5_K_M, ctx 131K, KV q8_0, mmproj F16 | ~26.5 GB |
| **this service** (large-v3-turbo `int8_float16`) | **+1.3 GB** |
| **Total on the V100-32G** | **~27.9 / 32 GB** (~4.9 GB headroom) |

## Gotchas (all hit in production, all fixed here)

1. **`av==16.0.1` is a hard pin.** faster-whisper 1.2.1 calls `av.open(..., metadata_errors="ignore")`; PyAV ≥ 17 removed that kwarg → every request dies with HTTP 500:
   `TypeError: open() got an unexpected keyword argument 'metadata_errors'`. A plain `pip install -r` without the pin resolves av 19 and silently breaks.
2. **The original model repo is gated.** `Systran/faster-whisper-large-v3-turbo` now returns **401 "Invalid username or password"** without a HuggingFace token. `server.py` therefore uses the public mirror **`h2oai/faster-whisper-large-v3-turbo`** (identical weights). To switch models, edit the `MODEL` constant — any faster-whisper repo name works (`Systran/faster-whisper-large-v3`, `-small`, …).
3. **cudnn/cublas come from pip wheels**, not the system: `nvidia-cudnn-cu12` / `nvidia-cublas-cu12` land in `<venv>/lib/pythonX.Y/site-packages/nvidia/`. The unit exports their `lib` dirs via `LD_LIBRARY_PATH` — if you run manually, export them too. **The path contains the Python version** (`python3.13`) — fix it for other Pythons:
   ```bash
   venv/bin/python -c "import nvidia.cudnn as c, nvidia.cublas as b, os; \
     print(os.path.dirname(c.__file__)+'/lib:'+os.path.dirname(b.__file__)+'/lib')"
   ```
4. **`TimeoutStartSec=900` is intentional.** First boot downloads ~1.6 GB from HF before the port opens; systemd's default 90 s kills it mid-download and `Restart=on-failure` loops forever. After caching, restarts take ~7 s.
5. **No auth, binds `0.0.0.0:18080`.** Fine on a trusted LAN (it is the V100 node's local API surface); otherwise firewall it or change `host` in `server.py`.
6. **CPU fallback:** set `device="cpu", compute_type="int8"` in `get_model()` — slower (~realtime) but no GPU needed.

## Model & performance notes

- `large-v3-turbo` ≈ `large-v3` quality at ~2× less VRAM and much higher speed; `int8_float16` halves VRAM again with negligible loss on clean speech.
- Measured on the V100: EN 11 s clip → **1.08 s** (~10× realtime); RU 2.9 s (espeak-ng robotic voice) → 0.36 s, language auto-detected (`ru`), lower confidence — expected for synthetic speech.
- `probability` in the response is whisper's language-detection confidence; clients (e.g. pi-v100-media) use it to filter silence/noise.

## systemd unit (reference)

```ini
[Unit]
Description=faster-whisper STT (large-v3-turbo int8_float16) on GPU0, port 18080
After=network-online.target nvidia-persistenced.service
Wants=network-online.target nvidia-persistenced.service

[Service]
WorkingDirectory=/root/whisper-svc
Environment=LD_LIBRARY_PATH=/root/whisper-svc/venv/lib/python3.13/site-packages/nvidia/cudnn/lib:/root/whisper-svc/venv/lib/python3.13/site-packages/nvidia/cublas/lib
TimeoutStartSec=900
ExecStart=/root/whisper-svc/venv/bin/python /root/whisper-svc/server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
