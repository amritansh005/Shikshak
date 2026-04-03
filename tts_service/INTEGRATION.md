# TTS Service — Integration Guide

## Service architecture position

```
Microphone → STT (port 9001)
                ↓ transcript + emotion_data
             techer_llm (port 8000)
                ↓ teacher_text + TeachingDirective
             tts_service (port 5000)   ← NEW
                ↓ WAV audio
             Speaker (sounddevice)
```

---

## 1. Start the TTS service

```bash
conda activate shikshak_tts
cd tts_service
cp .env.example .env        # edit model path / device as needed
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 5000 --workers 1
```

> **Backend note:** This setup uses **OpenVoice-lite (MeloTTS runtime)** by default.
> It is significantly lighter than large conversational TTS stacks and works on
> both GPU and CPU (GPU recommended for lower latency).
>
> If startup logs show `No module named 'melo'`, you're likely running uvicorn
> from a different Python environment. Re-activate the same env where `pip install`
> was executed and restart the service.

---

## 2. Wire into `voice_chat_client.py`

Add at the top of `voice_chat_client.py` (STT project):

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../tts_service"))
from tts_client import TTSClient

tts = TTSClient(tts_service_url="http://127.0.0.1:5000")
```

In `finalize_turn()`, after the `send_to_teacher()` call:

```python
teacher_text = send_to_teacher(session_id, final_text, emotion_data)
print(f"Teacher: {teacher_text}\n", flush=True)

# ── Speak the teacher response with emotion context ────────────────
tts.speak_with_emotion(
    text=teacher_text,
    emotion_data=emotion_data,    # same dict you sent to techer_llm
    session_id=session_id,
)
```

That's all — the TTSClient runs playback in a background thread so it
never blocks the STT pipeline.

---

## 3. Wire into `terminal_chat.py` (techer_llm project)

```python
import sys, os
sys.path.insert(0, "/path/to/tts_service")
from tts_client import TTSClient

tts = TTSClient()
```

After receiving the response from `/chat`:

```python
response_text = result["response"]
print(f"Teacher: {response_text}\n")
tts.speak_neutral(response_text)   # no emotion data in terminal mode
```

Or if you expose the TeachingDirective from the `/chat` response:

```python
# In techer_llm/app/main.py, add directive to ChatResponse:
#   class ChatResponse(BaseModel):
#       response: str
#       directive: Optional[dict] = None
#
# Then in terminal_chat.py:
directive_data = result.get("directive")
if directive_data:
    tts.speak(text=response_text, directive=SimpleNamespace(**directive_data))
else:
    tts.speak_neutral(response_text)
```

---

## 4. HTTP API reference

### POST /synthesize → audio/wav

```json
{
  "text": "Let me explain that differently...",
  "session_id": "voice-session-20250317-120000",
  "voice": "Chelsie",
  "emotion": {
    "smoothed_state": "confused",
    "smoothed_confidence": 0.72,
    "trend": "stable",
    "secondary_state": "anxious",
    "secondary_confidence": 0.18
  }
}
```

Response: `audio/wav` bytes + headers:
- `X-TTS-Latency-Ms`
- `X-TTS-Cache-Hit`
- `X-TTS-Resolved-State`
- `X-TTS-Backend`

### WebSocket /ws/synthesize

Send same JSON as above.
Receive: multiple binary PCM frames → final JSON `{"done": true, "latency_ms": ...}`.

### GET /health · GET /metrics · GET /voices

---

## 5. Emotion state → voice behaviour mapping

| Teaching State  | Rate   | Pitch   | Energy | Style                              |
|-----------------|--------|---------|--------|------------------------------------|
| `confused`      | −18%   | −0.5 st | 0.65   | Slow, clear, patient               |
| `frustrated`    | −12%   | −1.5 st | 0.62   | Gentle, warm, encouraging          |
| `anxious`       | −10%   | −1.0 st | 0.60   | Soft, grounded, reassuring         |
| `discouraged`   | −15%   | −2.0 st | 0.58   | Warm, empathetic, hopeful          |
| `uncertain`     | −8%    | −0.5 st | 0.65   | Steady, supportive                 |
| `bored`         | +8%    | +1.0 st | 0.78   | Energetic, lively                  |
| `confident`     | +5%    | +0.5 st | 0.75   | Bright, positive                   |
| `curious`       | ±0%    | +1.5 st | 0.78   | Inquisitive, exploratory           |
| `engaged`       | +5%    | +1.0 st | 0.80   | Enthusiastic, motivating           |
| `neutral`       | ±0%    | ±0 st   | 0.70   | Clear, friendly default            |

Trend modifiers applied on top:
- `escalating` → −4% rate, −5% energy
- `de-escalating` / `recovering` → +2–3% rate, +2–3% energy

---

## 6. Caching behaviour

The TTS service reuses your existing Redis instance (same `REDIS_URL`).
Cache keys are scoped to `(text, emotion_state, trend, voice)` so the same
phrase synthesised for a `confused` student sounds different from
the same phrase for an `engaged` one — they are stored as separate entries.

Short repeated phrases (greetings, confirmations, ≤200 chars) are cached for 1 hour.
Long responses are synthesised fresh each time.
