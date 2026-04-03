"""
Prosody controller for OpenVoice-lite TTS.

Maps TeachingDirective emotion signals (smoothed_state, trend, teaching_state,
secondary_state, negative_pressure) into:

  1. Voice parameters  — speaking rate, pitch shift, energy level
  2. Style prompt  — natural language emotional speaking guidance
     system-level speaking style hint
  3. SSML-style prosody hints  — injected around text for models that support
     <prosody> tags (gracefully skipped for models that do not)

Design principles:
  - Deterministic given same inputs (no randomness, fully reproducible)
  - No LLM call — pure rule-based mapping, runs in <1 ms
  - Mirrors the TeachingDirective structure from emotion_state_service.py
  - All numeric values are clamped to safe TTS ranges
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional


# ── Emotion → prosody parameter tables ───────────────────────────────────────
#
# Each entry defines the "centre" values for that teaching state.
# We linearly interpolate toward these targets based on smoothed_confidence.
# At confidence=0.0 we stay at NEUTRAL_PARAMS; at confidence=1.0 we hit the
# state target fully.

@dataclass(frozen=True)
class ProsodyParams:
    """Normalised prosody parameters.

    rate_multiplier : speaking speed relative to model default (1.0 = normal)
    pitch_shift_st  : semitones of pitch offset from model default (0 = flat)
    energy_level    : 0.0 (whisper) … 1.0 (loud/expressive)
    pause_after_sentence_ms : added pause between sentences (ms)
    """
    rate_multiplier: float = 1.0
    pitch_shift_st: float = 0.0
    energy_level: float = 0.7
    pause_after_sentence_ms: int = 200


# Baseline – used when confidence is low or state is neutral
_NEUTRAL = ProsodyParams(
    rate_multiplier=1.0,
    pitch_shift_st=0.0,
    energy_level=0.70,
    pause_after_sentence_ms=200,
)

_STATE_TARGETS: Dict[str, ProsodyParams] = {
    "neutral": _NEUTRAL,

    # ── Negative states ───────────────────────────────────────────
    "frustrated": ProsodyParams(
        rate_multiplier=0.88,   # slow down — give space
        pitch_shift_st=-1.5,    # warmer, lower
        energy_level=0.62,      # calmer energy
        pause_after_sentence_ms=350,
    ),
    "confused": ProsodyParams(
        rate_multiplier=0.82,   # noticeably slower
        pitch_shift_st=-0.5,
        energy_level=0.65,
        pause_after_sentence_ms=400,  # longer pauses let information settle
    ),
    "anxious": ProsodyParams(
        rate_multiplier=0.90,
        pitch_shift_st=-1.0,    # grounded, calm
        energy_level=0.60,      # soft, non-threatening
        pause_after_sentence_ms=320,
    ),
    "discouraged": ProsodyParams(
        rate_multiplier=0.85,
        pitch_shift_st=-2.0,    # warm, empathetic low
        energy_level=0.58,
        pause_after_sentence_ms=380,
    ),
    "uncertain": ProsodyParams(
        rate_multiplier=0.92,
        pitch_shift_st=-0.5,
        energy_level=0.65,
        pause_after_sentence_ms=300,
    ),

    # ── Neutral-ish ───────────────────────────────────────────────
    "bored": ProsodyParams(
        rate_multiplier=1.08,   # slightly faster — inject energy
        pitch_shift_st=1.0,     # slightly brighter
        energy_level=0.78,
        pause_after_sentence_ms=160,
    ),

    # ── Positive states ───────────────────────────────────────────
    "confident": ProsodyParams(
        rate_multiplier=1.05,
        pitch_shift_st=0.5,
        energy_level=0.75,
        pause_after_sentence_ms=180,
    ),
    "engaged": ProsodyParams(
        rate_multiplier=1.05,
        pitch_shift_st=1.0,
        energy_level=0.80,
        pause_after_sentence_ms=170,
    ),
    "curious": ProsodyParams(
        rate_multiplier=1.0,
        pitch_shift_st=1.5,     # higher ending inflection, inquisitive
        energy_level=0.78,
        pause_after_sentence_ms=190,
    ),
}

# trend modifier — applied on top of state target
_TREND_RATE_DELTA: Dict[str, float] = {
    "escalating": -0.04,       # escalating frustration → even slower
    "de-escalating": +0.03,    # things improving → slight lift
    "recovering": +0.02,
    "stable": 0.0,
}

_TREND_ENERGY_DELTA: Dict[str, float] = {
    "escalating": -0.05,
    "de-escalating": +0.03,
    "recovering": +0.02,
    "stable": 0.0,
}


# ── Style prompts (natural language speaking style guidance) ───────────────────
#
# These are retained as backend-agnostic emotional speaking instructions.
# Keep them short and imperative.

_STATE_STYLE_PROMPTS: Dict[str, str] = {
    "neutral": (
        "Speak clearly and naturally with a warm, friendly teaching tone."
    ),
    "frustrated": (
        "Speak slowly and gently with extra warmth and patience. "
        "Sound encouraging and calm, like a supportive tutor."
    ),
    "confused": (
        "Speak slowly and clearly, pausing between key ideas. "
        "Sound patient and careful, like explaining to someone who is working it out."
    ),
    "anxious": (
        "Speak softly and reassuringly, at an unhurried pace. "
        "Sound calm and grounded, like a trusted mentor."
    ),
    "discouraged": (
        "Speak warmly with genuine empathy. Sound hopeful and encouraging. "
        "Pause a little after important phrases."
    ),
    "uncertain": (
        "Speak gently and clearly. Sound steady and supportive, "
        "building the student's confidence with each sentence."
    ),
    "bored": (
        "Speak with energy and enthusiasm, slightly faster than normal. "
        "Sound lively and engaging, like sharing something exciting."
    ),
    "confident": (
        "Speak naturally and clearly with a bright, positive tone. "
        "Sound engaged and encouraging."
    ),
    "engaged": (
        "Speak with warmth and positive energy. Sound enthusiastic and motivating."
    ),
    "curious": (
        "Speak with an inquisitive, exploratory tone. "
        "Sound genuinely interested and encouraging of questions."
    ),
}

# Trend overlays — appended to base style prompt
_TREND_STYLE_SUFFIX: Dict[str, str] = {
    "escalating": " Take extra care to sound very calm and unhurried.",
    "de-escalating": " Sound gently uplifting, things are getting better.",
    "recovering": " Warmly acknowledge the progress the student is making.",
    "stable": "",
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolate from a toward b by t in [0, 1]."""
    return a + (b - a) * _clamp(t, 0.0, 1.0)


@dataclass
class ResolvedProsody:
    """Final prosody values + style prompt ready for the TTS engine."""

    # Numeric voice parameters
    rate_multiplier: float = 1.0
    pitch_shift_st: float = 0.0
    energy_level: float = 0.70
    pause_after_sentence_ms: int = 200

    # Natural language emotional speaking guidance
    style_prompt: str = "Speak clearly and naturally with a warm, friendly teaching tone."

    # Emotion context (informational — logged, not sent to model)
    resolved_state: str = "neutral"
    resolved_trend: str = "stable"
    smoothed_confidence: float = 0.5


def resolve_prosody(
    smoothed_state: str = "neutral",
    smoothed_confidence: float = 0.5,
    trend: str = "stable",
    secondary_state: str = "neutral",
    secondary_confidence: float = 0.0,
    negative_pressure: float = 0.0,
    rate_min: float = 0.75,
    rate_max: float = 1.30,
) -> ResolvedProsody:
    """Compute TTS prosody parameters from emotion state.

    Args:
        smoothed_state      : primary teaching state from EmotionStateService
        smoothed_confidence : confidence in smoothed_state (0–1)
        trend               : 'escalating' | 'de-escalating' | 'recovering' | 'stable'
        secondary_state     : secondary teaching state (used for blending)
        secondary_confidence: confidence in secondary state (0–1)
        negative_pressure   : scalar pressure score from EmotionStateService
        rate_min / rate_max : global rate clamp from settings

    Returns:
        ResolvedProsody with all parameters filled in.
    """
    primary_target = _STATE_TARGETS.get(smoothed_state, _NEUTRAL)
    secondary_target = _STATE_TARGETS.get(secondary_state, _NEUTRAL)

    # Blend primary and secondary based on confidences.
    # If secondary_confidence is low, we barely move from primary.
    blend_t = _clamp(secondary_confidence * 0.35, 0.0, 0.35)
    # Interpolate from primary toward neutral at low confidence.
    primary_t = _clamp(smoothed_confidence, 0.0, 1.0)

    raw_rate = _lerp(
        _lerp(_NEUTRAL.rate_multiplier, primary_target.rate_multiplier, primary_t),
        secondary_target.rate_multiplier,
        blend_t,
    )
    raw_pitch = _lerp(
        _lerp(_NEUTRAL.pitch_shift_st, primary_target.pitch_shift_st, primary_t),
        secondary_target.pitch_shift_st,
        blend_t,
    )
    raw_energy = _lerp(
        _lerp(_NEUTRAL.energy_level, primary_target.energy_level, primary_t),
        secondary_target.energy_level,
        blend_t,
    )
    raw_pause = _lerp(
        float(_lerp(float(_NEUTRAL.pause_after_sentence_ms), float(primary_target.pause_after_sentence_ms), primary_t)),
        float(secondary_target.pause_after_sentence_ms),
        blend_t,
    )

    # Apply trend modifiers
    rate_delta = _TREND_RATE_DELTA.get(trend, 0.0)
    energy_delta = _TREND_ENERGY_DELTA.get(trend, 0.0)

    final_rate = _clamp(raw_rate + rate_delta, rate_min, rate_max)
    final_pitch = _clamp(raw_pitch, -6.0, 6.0)
    final_energy = _clamp(raw_energy + energy_delta, 0.40, 1.0)
    final_pause = int(_clamp(raw_pause, 100, 600))

    # Style prompt
    base_style = _STATE_STYLE_PROMPTS.get(smoothed_state, _STATE_STYLE_PROMPTS["neutral"])
    trend_suffix = _TREND_STYLE_SUFFIX.get(trend, "")
    style_prompt = (base_style + trend_suffix).strip()

    return ResolvedProsody(
        rate_multiplier=round(final_rate, 3),
        pitch_shift_st=round(final_pitch, 2),
        energy_level=round(final_energy, 3),
        pause_after_sentence_ms=final_pause,
        style_prompt=style_prompt,
        resolved_state=smoothed_state,
        resolved_trend=trend,
        smoothed_confidence=round(smoothed_confidence, 3),
    )


# ── Sentence boundary splitting ───────────────────────────────────────────────
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries for streaming synthesis."""
    raw = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


def inject_sentence_pauses(text: str, pause_ms: int) -> str:
    """Insert SSML <break> tags between sentences if pause_ms > 0.

    For TTS engines that do not support SSML the raw text is returned unchanged
    — the caller should check model capabilities first.
    """
    if pause_ms <= 100:
        return text
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return text
    break_tag = f'<break time="{pause_ms}ms"/>'
    return f" {break_tag} ".join(sentences)
