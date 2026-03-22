# Shikshak Desktop App (PyQt6)

A thin desktop UI for your live AI teacher pipeline.

## What it does
- Keeps your existing backend/services unchanged
- Replaces the terminal client with a PyQt6 desktop app
- Preserves live microphone conversation, VAD, partial/final STT, teacher calls, TTS playback, interruption handling, and speaker verification

## Expected project layout
Place this folder beside your existing projects:

```text
Shikshak/
├── desktop_teacher_app/
├── STT/
├── techer_llm/
└── tts_service/
```

## Run
```bash
cd desktop_teacher_app
pip install -r requirements.txt
python main.py
```

## Notes
- The app imports and reuses your existing STT / TTS / teacher service code.
- It does not modify backend behavior.
- Backend URLs and audio thresholds still come from your existing `.env` / config files where possible.
