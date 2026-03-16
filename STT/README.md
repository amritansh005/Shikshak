# STT voice pipeline

This package now uses a stronger phase-2.5 end-of-turn stack:

Mic -> raw VAD -> partial Whisper transcript -> turn manager -> final transcript -> teacher chatbot

## What changed

The turn manager no longer ends a turn just because a fixed silence happened.
It now combines several signals:

- soft pause detection
- resume window
- incomplete phrase holding
- transcript stability checks
- filler / non-meaningful utterance filtering
- semantic completion scoring
- hard silence fallback

## Simple behavior

If the user says:

"Photosynthesis is the process ..."

and pauses briefly, the system will usually keep waiting because the phrase looks incomplete.

If the user says:

"Can you explain Newton's first law"

and then pauses, the system will finalize sooner because the question looks complete and stable.

## Tuning

Adjust these in `.env`:

- `TURN_SOFT_SILENCE_MS`
- `TURN_RESUME_WINDOW_MS`
- `TURN_INCOMPLETE_HOLD_MS`
- `TURN_UNSTABLE_HOLD_MS`
- `TURN_SEMANTIC_HOLD_MS`
- `TURN_FORCE_STABLE_FINALIZE_MS`
- `TURN_HARD_SILENCE_MS`
- `TURN_STABLE_WAIT_MS`
- `TURN_COMPLETION_SCORE_FINALIZE`

## Current scope

This version improves pause handling and semantic endpointing.
It does not yet include:

- TTS-aware interruption
- echo cancellation from assistant playback
- full duplex speaker/mic control
