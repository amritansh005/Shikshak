"""
Emotion detection from text, audio prosody, and SER model.

Three independent signals:
1. Text emotion — keyword/pattern classifier on the transcript.
2. Prosodic features — speech rate, pause ratio, pitch proxy from raw PCM.
3. SER model — emotion2vec_plus_base via FunASR for audio emotion classification.
   Runs on CPU (per-turn inference after the turn ends).

No LLM is used here. These are fast, dedicated classifiers that run
in the STT process alongside Whisper.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── SER Model (emotion2vec_plus_base via FunASR) ──────────────────

# 9-class labels from emotion2vec_plus_base.
SER_LABELS = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "other",
    6: "sad",
    7: "surprised",
    8: "unknown",
}

SER_MODEL_ID = os.getenv("SER_MODEL", "iic/emotion2vec_plus_base")


class SERModel:
    """Speech Emotion Recognition using emotion2vec_plus_base.

    ~90M parameters. Loaded once at startup via FunASR.
    Runs on CPU per-turn (not real-time).
    Accepts 16kHz mono audio as numpy float32 array.
    """

    def __init__(self) -> None:
        from funasr import AutoModel as FunASRAutoModel

        logger.info("Loading SER model | model=%s | device=cpu", SER_MODEL_ID)

        self.model = FunASRAutoModel(
            model=SER_MODEL_ID,
            device="cpu",
        )

        logger.info("SER model loaded | model=%s", SER_MODEL_ID)

    def classify(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, object]:
        """Classify emotion from raw audio.

        Args:
            audio: float32 numpy array, mono, 16kHz.
            sample_rate: Must be 16000.

        Returns:
            {
                "label": str,           # e.g. "angry", "happy", "sad", "neutral"
                "confidence": float,     # 0.0 to 1.0
                "all_scores": dict,      # all class probabilities
            }
        """
        if audio is None or len(audio) == 0:
            return {"label": "neutral", "confidence": 0.5, "all_scores": {}}

        try:
            res = self.model.generate(
                input=audio,
                granularity="utterance",
                extract_embedding=False,
            )

            # FunASR returns a list of dicts. First element has 'labels' and 'scores'.
            if not res or not isinstance(res, list) or len(res) == 0:
                return {"label": "neutral", "confidence": 0.5, "all_scores": {}}

            result = res[0]
            labels = result.get("labels", [])
            scores = result.get("scores", [])

            if not labels or not scores:
                return {"label": "neutral", "confidence": 0.5, "all_scores": {}}

            # Find the top label.
            # labels is a list like ['/angry', '/happy', ...] or just label strings.
            # scores is the corresponding probabilities.
            all_scores = {}
            best_label = "neutral"
            best_score = 0.0

            for lbl, score in zip(labels, scores):
                # Clean label (FunASR sometimes prefixes with '/').
                clean_label = str(lbl).strip("/").strip().lower()
                score_val = float(score)
                all_scores[clean_label] = round(score_val, 4)

                if score_val > best_score:
                    best_score = score_val
                    best_label = clean_label

            return {
                "label": best_label,
                "confidence": best_score,
                "all_scores": all_scores,
            }

        except Exception as exc:
            logger.warning("SER model inference failed | error=%s", exc)
            return {"label": "neutral", "confidence": 0.5, "all_scores": {}}


# ── Text-based emotion classification ──────────────────────────────

# Each pattern list maps to an emotion label + base confidence.
# Patterns are checked in priority order; first match wins.
# This is deliberately simple — a fine-tuned BERT model would be
# better but this gives us a working baseline with zero dependencies.

_TEXT_EMOTION_RULES: List[Dict] = [
    {
        "label": "frustrated",
        "patterns": [
            r"\bi\s*(still\s+)?don'?t\s+(get|understand)\b",
            r"\bthis\s+(is\s+)?(so\s+)?(hard|difficult|confusing|complicated)\b",
            r"\bnothing\s+makes\s+sense\b",
            r"\bi\s+give\s+up\b",
            r"\bwhat\s+even\b",
            r"\bthis\s+is\s+(too|very)\s+(much|hard)\b",
            r"\bi\s+can'?t\s+(do|figure|understand|solve)\b",
            r"\bugh\b",
            r"\bwhy\s+is\s+this\s+so\b",
            r"\bi\s+keep\s+getting\s+(it\s+)?wrong\b",
        ],
        "confidence": 0.75,
    },
    {
        "label": "confused",
        "patterns": [
            r"\bi\s+don'?t\s+(understand|get\s+it)\b",
            r"\bwhat\s+do\s+you\s+mean\b",
            r"\bi'?m\s+(confused|lost)\b",
            r"\bcan\s+you\s+(explain|say)\s+(that\s+)?again\b",
            r"\bhow\s+(does|is)\s+that\s+(work|possible)\b",
            r"\bwait\s+what\b",
            r"\bthat\s+doesn'?t\s+make\s+sense\b",
            r"\bwhat\s+is\s+the\s+difference\b",
            r"\bhuh\b",
            r"\bwhy\s+(does|is|do)\b",
        ],
        "confidence": 0.70,
    },
    {
        "label": "bored",
        "patterns": [
            r"\bthis\s+is\s+(boring|dull)\b",
            r"\bi\s+(already\s+)?know\s+(this|that|all\s+this)\b",
            r"\bcan\s+we\s+(move\s+on|skip)\b",
            r"\bok\s*(ay)?\s*\.?\s*$",
            r"\byeah\s*(\.?\s*)$",
            r"\bwhatever\b",
            r"\btoo\s+(easy|simple|basic)\b",
            r"\bi\s+know\s+i\s+know\b",
        ],
        "confidence": 0.60,
    },
    {
        "label": "confident",
        "patterns": [
            r"\bi\s+(get|got|understand)\s+(it|this|that)\b",
            r"\boh\s+(i\s+see|ok|okay|that\s+makes\s+sense)\b",
            r"\bthat\s+makes\s+sense\b",
            r"\bi\s+think\s+i\s+(understand|got\s+it)\b",
            r"\beasy\b",
            r"\bi\s+can\s+do\s+(this|that|it)\b",
            r"\baha\b",
            r"\bnow\s+i\s+(get|understand)\b",
        ],
        "confidence": 0.65,
    },
    {
        "label": "curious",
        "patterns": [
            r"\bbut\s+what\s+(if|about|happens)\b",
            r"\bwhat\s+would\s+happen\b",
            r"\bcan\s+you\s+tell\s+me\s+more\b",
            r"\bthat'?s\s+interesting\b",
            r"\bwow\b",
            r"\breally\b\?",
            r"\bhow\s+come\b",
            r"\bwhat\s+else\b",
            r"\btell\s+me\s+more\b",
        ],
        "confidence": 0.60,
    },
    {
        "label": "anxious",
        "patterns": [
            r"\bi'?m\s+(scared|nervous|worried)\s+(about|of|for)\b",
            r"\bwhat\s+if\s+i\s+(fail|get\s+it\s+wrong|can'?t)\b",
            r"\bis\s+this\s+(going\s+to\s+be\s+)?(on|in)\s+the\s+(exam|test)\b",
            r"\bi'?m\s+not\s+(sure|ready|good\s+enough)\b",
            r"\bthis\s+is\s+stressful\b",
        ],
        "confidence": 0.65,
    },
]


def classify_text_emotion(text: str) -> Dict[str, object]:
    """Classify emotion from transcript text using pattern matching.

    Returns:
        {
            "label": str,        # e.g. "frustrated", "confused", "neutral"
            "confidence": float,  # 0.0 to 1.0
        }
    """
    if not text or not text.strip():
        return {"label": "neutral", "confidence": 0.5}

    lowered = text.lower().strip()

    for rule in _TEXT_EMOTION_RULES:
        for pattern in rule["patterns"]:
            if re.search(pattern, lowered):
                return {
                    "label": rule["label"],
                    "confidence": rule["confidence"],
                }

    return {"label": "neutral", "confidence": 0.5}


# ── Prosodic feature extraction from raw PCM ──────────────────────

@dataclass
class ProsodyFeatures:
    """Prosodic features extracted from a single turn's audio."""

    speech_rate_syllables_per_sec: float = 0.0
    pause_ratio: float = 0.0          # fraction of turn that is silence
    mean_pause_duration_ms: float = 0.0
    filled_pause_count: int = 0        # "um", "uh" etc from transcript
    pitch_mean_hz: float = 0.0
    pitch_std_hz: float = 0.0
    energy_mean: float = 0.0
    energy_std: float = 0.0
    total_speech_duration_sec: float = 0.0
    total_silence_duration_sec: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "speech_rate_sps": round(self.speech_rate_syllables_per_sec, 2),
            "pause_ratio": round(self.pause_ratio, 3),
            "mean_pause_ms": round(self.mean_pause_duration_ms, 1),
            "filled_pauses": self.filled_pause_count,
            "pitch_mean_hz": round(self.pitch_mean_hz, 1),
            "pitch_std_hz": round(self.pitch_std_hz, 1),
            "energy_mean": round(self.energy_mean, 4),
            "energy_std": round(self.energy_std, 4),
        }


# Rough syllable estimator for English: count vowel clusters.
_SYLLABLE_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)
_FILLED_PAUSE_RE = re.compile(r"\b(uh|um|umm|uhh|hmm|er|erm|mm|mmm)\b", re.IGNORECASE)


def _estimate_syllables(text: str) -> int:
    words = text.split()
    total = 0
    for w in words:
        matches = _SYLLABLE_RE.findall(w)
        total += max(len(matches), 1)
    return total


def _estimate_pitch_from_autocorrelation(
    audio: np.ndarray,
    sample_rate: int,
    frame_size: int = 1024,
    hop_size: int = 512,
    f_min: float = 75.0,
    f_max: float = 500.0,
) -> List[float]:
    """Simple autocorrelation-based pitch estimator.

    Not as accurate as CREPE or WORLD but runs fast with zero
    dependencies beyond numpy.
    """
    pitches: List[float] = []
    min_lag = int(sample_rate / f_max)
    max_lag = int(sample_rate / f_min)

    for start in range(0, len(audio) - frame_size, hop_size):
        frame = audio[start : start + frame_size]

        # Skip low-energy frames (silence).
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 0.01:
            continue

        # Normalized autocorrelation.
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2 :]

        if max_lag >= len(corr):
            continue

        search = corr[min_lag : max_lag + 1]
        if len(search) == 0:
            continue

        peak_idx = np.argmax(search) + min_lag
        if peak_idx > 0 and corr[peak_idx] > 0.3 * corr[0]:
            pitches.append(sample_rate / peak_idx)

    return pitches


def extract_prosody(
    pcm_frames: List[bytes],
    is_speech_flags: List[bool],
    transcript: str,
    sample_rate: int = 16000,
    frame_ms: int = 30,
) -> ProsodyFeatures:
    """Extract prosodic features from a completed turn.

    Args:
        pcm_frames: List of raw PCM16 audio frames from the turn.
        is_speech_flags: Per-frame VAD speech/silence flag.
        transcript: Final transcript text for the turn.
        sample_rate: Audio sample rate.
        frame_ms: Duration of each frame in milliseconds.
    """
    features = ProsodyFeatures()

    if not pcm_frames or not transcript.strip():
        return features

    total_frames = len(pcm_frames)
    speech_frames = sum(1 for f in is_speech_flags if f)
    silence_frames = total_frames - speech_frames

    features.total_speech_duration_sec = speech_frames * frame_ms / 1000.0
    features.total_silence_duration_sec = silence_frames * frame_ms / 1000.0
    total_duration = total_frames * frame_ms / 1000.0

    if total_duration > 0:
        features.pause_ratio = features.total_silence_duration_sec / total_duration

    # Mean pause duration: count contiguous silence runs.
    pause_runs: List[int] = []
    current_run = 0
    for is_speech in is_speech_flags:
        if not is_speech:
            current_run += 1
        else:
            if current_run > 0:
                pause_runs.append(current_run)
            current_run = 0
    if current_run > 0:
        pause_runs.append(current_run)

    if pause_runs:
        features.mean_pause_duration_ms = (
            sum(pause_runs) * frame_ms / len(pause_runs)
        )

    # Speech rate (syllables per second of speech, not total duration).
    syllables = _estimate_syllables(transcript)
    if features.total_speech_duration_sec > 0.5:
        features.speech_rate_syllables_per_sec = (
            syllables / features.total_speech_duration_sec
        )

    # Filled pauses from transcript.
    features.filled_pause_count = len(_FILLED_PAUSE_RE.findall(transcript))

    # Combine all frames into a single float32 audio array for pitch/energy.
    try:
        all_bytes = b"".join(pcm_frames)
        audio = np.frombuffer(all_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Energy.
        if len(audio) > 0:
            features.energy_mean = float(np.mean(np.abs(audio)))
            features.energy_std = float(np.std(np.abs(audio)))

        # Pitch estimation.
        pitches = _estimate_pitch_from_autocorrelation(audio, sample_rate)
        if pitches:
            features.pitch_mean_hz = float(np.mean(pitches))
            features.pitch_std_hz = float(np.std(pitches))

    except Exception as exc:
        logger.warning("Prosody extraction failed for audio | error=%s", exc)

    return features
