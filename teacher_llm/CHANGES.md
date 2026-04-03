# Single-Model Memory Card Extraction — Changes

## Overview

Eliminated the second VRAM model (qwen2.5:0.5b). **Only gemma3:4b** now runs in VRAM.
Memory card extraction happens **inline during TTS playback** using the same Gemma model,
since the GPU is idle while TTS is playing audio.

## What Changed

### `.env`
- `BG_LLM_MODEL` changed from `qwen2.5:0.5b` → `gemma3:4b`
- You can now run Ollama with `OLLAMA_MAX_LOADED_MODELS=1`

### `app/services/llm_service.py`
- **`fg_model_lock`** (threading.Lock) — serialises access to the foreground model between teacher response generation and memory card extraction. Prevents Ollama request overlap.
- **`_memory_card_cancel`** (threading.Event) — signals in-flight memory card extraction to abort when a new student turn arrives.
- **`generate()`** — now sets the cancel event and acquires `fg_model_lock` before calling Ollama. Clears the cancel event once it holds the lock.
- **`stream_generate()`** — same lock/cancel treatment.
- **`foreground_structured_chat()`** — NEW method. Like `structured_chat()` but uses the foreground model (Gemma 4b) with `fg_model_lock` and cancellation checks before, during, and after the Ollama call.
- **`cancel_memory_card_extraction()`** — NEW helper. Sets the cancel event.
- `bg_model` defaults to `self.model` (Gemma 4b) instead of `gemma2:2b`.
- `bg_client` defaults to the same host as `fg_client`.

### `app/services/memory_card_service.py`
- **`extract_and_store_inline()`** — NEW method. Called from the `/chat` endpoint's background thread. Uses `foreground_structured_chat()` instead of `structured_chat()`. Checks `_memory_card_cancel` before embedding and DB writes. Marks messages as `memory_extracted=1` after completion (so `memory_worker` doesn't double-process).
- All existing methods (`_extract_and_store`, `extract_and_store_memory_card_from_messages`) are unchanged — `memory_worker.py` still works as a fallback.

### `app/main.py`
- Imports and initialises `MemoryCardService` in the lifespan.
- **`/chat` endpoint** now:
  1. Calls `_llm.cancel_memory_card_extraction()` at the top (step 0) — aborts any in-flight memory card extraction from the previous turn.
  2. After saving messages, fires `_extract_memory_card_background()` in a daemon thread alongside the existing summary update thread.
- Model warmup simplified (single model).

### `memory_worker.py`
- Updated docstring: now marked as **optional fallback**. With inline extraction, you don't need to run it. But if you do, it catches any turns that were cancelled before completion (acts as a safety net).

## Edge Cases Handled

### 1. User interrupts during TTS → new turn arrives while memory card is being extracted
- The new `/chat` call immediately sets `_memory_card_cancel`.
- `generate()` then sets it again and waits for `fg_model_lock`.
- Inside `foreground_structured_chat()`, the cancel event is checked:
  - **Before acquiring lock** → skips immediately
  - **After acquiring lock** → skips immediately  
  - **After Ollama returns** → discards the result
- The lock is released, `generate()` acquires it, and the teacher response proceeds with zero wasted time (beyond whatever Ollama already computed).

### 2. TTS finishes, user starts speaking, but memory card extraction is still running
- `finalize_turn()` calls `send_to_teacher()` → `/chat` endpoint.
- The `/chat` endpoint immediately calls `cancel_memory_card_extraction()`.
- `generate()` sets the cancel event and tries to acquire `fg_model_lock`.
- If the memory card Ollama call hasn't returned yet, `generate()` will block on the lock until it finishes (Ollama processes sequentially per model). But the cancel event ensures `foreground_structured_chat()` skips all post-processing (embedding, DB writes) and releases the lock as fast as possible.
- **Worst case latency**: the time remaining on the in-flight Ollama inference for memory card extraction. With `num_ctx=512` this is typically < 1 second.

### 3. Memory card extraction completes normally (no interruption, TTS finished)
- The background thread runs to completion, stores the memory card in SQLite/Redis, marks messages as extracted, and exits. No impact on anything.

### 4. Memory card extraction cancelled → messages left unprocessed
- If cancelled, the messages keep `memory_extracted=0` in SQLite. If `memory_worker.py` is running as a fallback, it will eventually pick them up. If not, they're simply not processed — memory cards are best-effort and not critical for the next turn.

## Interruption System — NOT Affected

The interruption pipeline is entirely CPU-based:
- **VAD**: Silero (CPU)
- **Noise gate**: CPU
- **Partial ASR**: Whisper STT (CPU)
- **TTS control**: stop/duck/restore/resume (CPU + audio device)

None of these touch Ollama or the GPU. The `fg_model_lock` only gates Ollama API calls, not the VAD/ASR event loop. The main event loop in `live_session_worker.py` continues processing speech frames at all times, regardless of what Ollama is doing.

## VRAM Savings

Before: gemma3:4b (~2.5 GiB) + qwen2.5:0.5b (~0.4 GiB) + KV caches ≈ 3.5-4.3 GiB
After:  gemma3:4b (~2.5 GiB) + one KV cache ≈ 2.7 GiB

~1-1.5 GiB freed.
