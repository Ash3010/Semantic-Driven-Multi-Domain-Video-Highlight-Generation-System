# Multi‑Domain Video Highlights + Semantic Summary (Demo)

A demo-ready project that generates **highlights** and a **semantic summary** from **any video** (sports, lectures, generic).

## What it does
- Upload a video (MP4/MOV) or provide a YouTube link
- Picks key segments using a **multi-signal scoring pipeline**:
  - Scene detection (PySceneDetect)
  - Audio energy peaks (librosa)
  - **Semantic importance** from transcript embeddings (Sentence‑BERT)
- Exports:
  - `highlights.mp4`
  - `timestamps.json` (why each moment was chosen)
  - a readable summary (bullets + paragraph)

---

## 1) Setup

### Prerequisites
- Python 3.10+ recommended
- **ffmpeg** installed and available in PATH

Check:
```bash
ffmpeg -version
```

### Create env + install
```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows (powershell)
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

> First run will download pretrained models (Whisper + Sentence‑BERT). That’s expected.

---

## 2) Run the UI
```bash
streamlit run app.py
```

---

## 3) Run from CLI (optional)
```bash
python -m pipeline.cli   --video "path/to/video.mp4"   --mode lecture   --target_seconds 75
```

Outputs go to: `outputs/<run_id>/`

---

## Troubleshooting
- If export fails: ensure `ffmpeg` is installed.
- If transcription is slow: choose `tiny` model in the UI.
- If video has no speech: the semantic module falls back and uses audio/visual cues.
