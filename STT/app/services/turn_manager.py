from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List

from app.config import settings


@dataclass
class PartialSnapshot:
    text: str
    timestamp: float
    audio_seconds: float


@dataclass
class TurnDecision:
    action: str  # continue | finalize | discard
    reason: str


@dataclass
class TurnState:
    frames: List[bytes] = field(default_factory=list)
    partial_history: List[PartialSnapshot] = field(default_factory=list)
    speech_active: bool = False
    total_audio_seconds: float = 0.0
    trailing_silence_ms: int = 0
    voiced_frames: int = 0
    silence_frames: int = 0
    last_voice_timestamp: float = 0.0
    speech_start_timestamp: float = 0.0
    soft_end_announced: bool = False
    soft_end_timestamp: float = 0.0

    def reset(self) -> None:
        self.frames.clear()
        self.partial_history.clear()
        self.speech_active = False
        self.total_audio_seconds = 0.0
        self.trailing_silence_ms = 0
        self.voiced_frames = 0
        self.silence_frames = 0
        self.last_voice_timestamp = 0.0
        self.speech_start_timestamp = 0.0
        self.soft_end_announced = False
        self.soft_end_timestamp = 0.0

    def snapshot_audio(self) -> bytes:
        """Return a copy of all accumulated PCM frames as a single bytes object.
        
        NOTE: The caller MUST hold the lock that guards this TurnState
        before calling this method, since self.frames is mutated by the
        main loop thread.
        """
        return b"".join(list(self.frames))

    def latest_partial(self) -> str:
        return self.partial_history[-1].text if self.partial_history else ""


class TurnManager:
    """Pause-tolerant end-of-turn manager.

    This sits above raw VAD and decides whether a silence is:
    - a natural thinking pause
    - an unfinished phrase
    - a stable completed turn
    - junk / filler that should be discarded

    It intentionally avoids LLM calls for endpointing so the pipeline stays fast.
    """

    # ── Per-language word lists ──────────────────────────────────────────
    _LANG = {
        "en": {
            "incomplete_endings": {
                "and", "or", "but", "so", "because", "if", "then", "than", "to", "for",
                "of", "in", "on", "at", "with", "from", "into", "about", "like", "that",
                "which", "who", "whom", "whose", "when", "where", "while", "as", "by",
                "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                "do", "does", "did", "can", "could", "should", "would", "will", "please",
                "uh", "um", "hmm", "actually", "also", "first", "second", "third", "explain",
                "tell", "show", "give", "what", "why", "how", "where", "when", "whether",
                "this", "that", "these", "those", "my", "your", "our", "their", "his", "her",
                "more", "another", "about", "regarding", "means", "meaning", "called",
            },
            "filler_words": {
                "uh", "um", "hmm", "huh", "ah", "er", "erm", "mm", "mmm", "like", "so",
                "well", "okay", "ok", "wait", "actually", "basically", "you", "know",
            },
            "non_meaningful_phrases": {
                "uh", "um", "hmm", "huh", "ah", "er", "erm", "mm", "okay", "ok", "wait",
                "one second", "just a second", "hold on", "let me think",
            },
            "question_starters": {
                "what", "why", "how", "when", "where", "who", "whom", "whose", "which", "can",
                "could", "would", "will", "should", "is", "are", "do", "does", "did", "tell",
                "explain", "show", "teach", "help", "please",
            },
        },
        "hi": {
            "incomplete_endings": {
                "aur", "ya", "lekin", "par", "kyunki", "agar", "toh", "phir", "ke", "ki",
                "ka", "ko", "se", "mein", "par", "tak", "ne", "hai", "hain", "tha", "the",
                "thi", "ho", "hoga", "hogi", "honge", "kya", "kaise", "kahan", "kab", "kaun",
                "konsa", "konsi", "jaise", "matlab", "woh", "yeh", "ye", "mera", "meri",
                "tumhara", "tumhari", "hamara", "hamari", "unka", "unki", "iska", "iski",
                "samjhao", "batao", "dikhao", "pehle", "doosra", "teesra",
            },
            "filler_words": {
                "uh", "um", "hmm", "huh", "ah", "er", "mm", "mmm", "haan", "achha",
                "theek", "bas", "matlab", "actually", "basically", "like", "so",
                "dekho", "suno", "wait", "ruko",
            },
            "non_meaningful_phrases": {
                "uh", "um", "hmm", "huh", "ah", "er", "mm", "haan", "achha", "theek",
                "bas", "ruko", "ek second", "ek minute", "sochne do", "wait",
            },
            "question_starters": {
                "kya", "kyun", "kaise", "kab", "kahan", "kaun", "konsa", "konsi",
                "batao", "samjhao", "dikhao", "sikhao", "bolo", "help",
            },
        },
    }

    def __init__(self) -> None:
        self.frame_ms = settings.audio_frame_ms

        lang = settings.whisper_language
        lang_data = self._LANG.get(lang, self._LANG["en"])

        self.INCOMPLETE_ENDINGS: set[str] = lang_data["incomplete_endings"]
        self.FILLER_WORDS: set[str] = lang_data["filler_words"]
        self.NON_MEANINGFUL_PHRASES: set[str] = lang_data["non_meaningful_phrases"]
        self.QUESTION_STARTERS: set[str] = lang_data["question_starters"]

    def start_turn(self, state: TurnState) -> None:
        state.reset()
        state.speech_active = True
        now = time.monotonic()
        state.speech_start_timestamp = now
        state.last_voice_timestamp = now

    def append_frame(self, state: TurnState, frame: bytes, *, is_speech: bool) -> float:
        state.frames.append(frame)
        state.total_audio_seconds = len(state.frames) * self.frame_ms / 1000.0
        if is_speech:
            state.voiced_frames += 1
            state.trailing_silence_ms = 0
            state.last_voice_timestamp = time.monotonic()
            state.soft_end_announced = False
            state.soft_end_timestamp = 0.0
        else:
            state.silence_frames += 1
            state.trailing_silence_ms += self.frame_ms
        return state.total_audio_seconds

    def register_partial(self, state: TurnState, text: str, audio_seconds: float) -> None:
        cleaned = self._normalize(text)
        if not cleaned:
            return
        now = time.monotonic()
        if state.partial_history and state.partial_history[-1].text == cleaned:
            state.partial_history[-1].timestamp = now
            state.partial_history[-1].audio_seconds = audio_seconds
            return
        state.partial_history.append(
            PartialSnapshot(text=cleaned, timestamp=now, audio_seconds=audio_seconds)
        )
        if len(state.partial_history) > 12:
            state.partial_history = state.partial_history[-12:]

    def should_attempt_partial(self, state: TurnState, last_partial_audio_seconds: float) -> bool:
        current_audio_seconds = state.total_audio_seconds
        enough_new_audio = (
            current_audio_seconds - last_partial_audio_seconds
            >= settings.partial_decode_interval_ms / 1000.0
        )
        if current_audio_seconds < settings.partial_min_audio_ms / 1000.0:
            return False
        if not enough_new_audio:
            return False
        if state.trailing_silence_ms >= settings.turn_hard_silence_ms:
            return False
        return True

    def evaluate(self, state: TurnState) -> TurnDecision:
        if state.total_audio_seconds < settings.whisper_min_audio_ms / 1000.0:
            if state.trailing_silence_ms >= settings.turn_hard_silence_ms:
                return TurnDecision("discard", "too_short")
            return TurnDecision("continue", "short_wait")

        latest = state.latest_partial()
        silence = state.trailing_silence_ms
        stable = self._is_stable(state)
        meaningful = self._is_meaningful(latest)
        incomplete = self._looks_incomplete(latest)
        complete = self._looks_complete(latest)
        likely_filler = self._is_likely_filler_turn(latest)
        semantic_score = self._completion_score(latest)

        if silence >= settings.turn_hard_silence_ms:
            if not meaningful or likely_filler:
                return TurnDecision("discard", "hard_silence_filler")
            return TurnDecision("finalize", "hard_silence")

        if silence < settings.turn_soft_silence_ms:
            return TurnDecision("continue", "pause_too_short")

        now = time.monotonic()
        if not state.soft_end_announced:
            state.soft_end_announced = True
            state.soft_end_timestamp = now
            return TurnDecision("continue", "soft_pause_resume_window")

        wait_in_soft_pause_ms = int((now - state.soft_end_timestamp) * 1000) if state.soft_end_timestamp else 0

        if not latest:
            if silence >= settings.turn_resume_window_ms:
                return TurnDecision("discard", "silence_without_text")
            return TurnDecision("continue", "awaiting_text")

        # Layer 1: discard tiny filler-like turns.
        if likely_filler and silence >= settings.turn_resume_window_ms:
            return TurnDecision("discard", "filler_pause")

        # Layer 2: strong complete turn.
        if complete and stable and wait_in_soft_pause_ms >= settings.turn_stable_wait_ms:
            return TurnDecision("finalize", "complete_and_stable")

        # Layer 3: short natural pause on incomplete phrasing. Keep waiting.
        if incomplete and silence < settings.turn_incomplete_hold_ms:
            return TurnDecision("continue", "incomplete_phrase_hold")

        # Layer 4: if transcript is still changing recently, wait a bit.
        if not stable and silence < settings.turn_unstable_hold_ms:
            return TurnDecision("continue", "transcript_unstable")

        # Layer 5: semantically likely complete after enough pause.
        if meaningful and semantic_score >= settings.turn_completion_score_finalize and silence >= settings.turn_resume_window_ms:
            return TurnDecision("finalize", "semantic_completion")

        # Layer 6: meaningful but not perfect — finalize after a longer hold.
        if meaningful and silence >= settings.turn_semantic_hold_ms and not incomplete:
            return TurnDecision("finalize", "semantic_hold_elapsed")

        # Layer 7: if the user paused a long time and transcript has stopped changing, finalize.
        if meaningful and stable and silence >= settings.turn_force_stable_finalize_ms:
            return TurnDecision("finalize", "stable_long_pause")

        return TurnDecision("continue", "wait_for_more_context")

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"\b[\w']+\b", text))

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[\w']+\b", text.lower())

    def _is_meaningful(self, text: str) -> bool:
        if not text:
            return False
        normalized = text.lower().strip()
        if normalized in self.NON_MEANINGFUL_PHRASES:
            return False
        if self._word_count(text) >= settings.turn_min_meaningful_words:
            return True
        if text.endswith(("?", ".", "!")) and self._word_count(text) >= 1:
            return True
        return False

    def _looks_complete(self, text: str) -> bool:
        if not text:
            return False
        if text.endswith(("?", ".", "!")):
            return True
        tokens = self._tokenize(text)
        if not tokens:
            return False
        if tokens[0] in self.QUESTION_STARTERS and self._word_count(text) >= 4 and not self._looks_incomplete(text):
            return True
        if self._word_count(text) >= settings.turn_complete_sentence_min_words and not self._looks_incomplete(text):
            return True
        return False

    def _looks_incomplete(self, text: str) -> bool:
        if not text:
            return True
        lowered = text.lower().strip()
        if lowered.endswith((",", ":", ";", "-", "—", "(", "/")):
            return True
        words = self._tokenize(lowered)
        if not words:
            return True
        last = words[-1]
        if last in self.INCOMPLETE_ENDINGS:
            return True
        if len(words) <= 2 and not lowered.endswith(("?", ".", "!")):
            return True
        if len(words) >= 2 and words[-2] in {"the", "a", "an", "my", "your", "our"}:
            return True
        if re.search(r"\b(and|or|but|because|if|so)\s*$", lowered):
            return True
        open_parens = lowered.count("(") > lowered.count(")")
        quotes_odd = lowered.count('"') % 2 == 1
        return open_parens or quotes_odd

    def _is_likely_filler_turn(self, text: str) -> bool:
        normalized = text.lower().strip()
        if not normalized:
            return True
        if normalized in self.NON_MEANINGFUL_PHRASES:
            return True
        words = self._tokenize(normalized)
        if not words:
            return True
        if len(words) <= 2 and all(w in self.FILLER_WORDS for w in words):
            return True
        return False

    def _completion_score(self, text: str) -> float:
        if not text:
            return 0.0

        score = 0.0
        words = self._word_count(text)
        lowered = text.lower().strip()

        if text.endswith(("?", ".", "!")):
            score += 0.45
        if words >= settings.turn_complete_sentence_min_words:
            score += 0.25
        elif words >= 3:
            score += 0.12
        if not self._looks_incomplete(text):
            score += 0.20
        if self._is_meaningful(text):
            score += 0.10
        if self._tokenize(lowered)[:1] and self._tokenize(lowered)[0] in self.QUESTION_STARTERS and words >= 4:
            score += 0.10
        return min(score, 1.0)

    def _is_stable(self, state: TurnState) -> bool:
        history = state.partial_history
        if len(history) < 2:
            return False

        newest = history[-1]
        previous = history[-2]
        if newest.text == previous.text:
            age_ms = (time.monotonic() - newest.timestamp) * 1000
            return age_ms >= settings.turn_stable_wait_ms

        a_words = self._tokenize(newest.text)
        b_words = self._tokenize(previous.text)
        if not a_words or not b_words:
            return False

        # Count how many words from the shorter list appear in the longer.
        # This handles Whisper prepending/appending a word between partials
        # (e.g. "explain photosynthesis" vs "can you explain photosynthesis").
        shorter, longer = (b_words, a_words) if len(b_words) <= len(a_words) else (a_words, b_words)
        longer_set = set(longer)
        common = sum(1 for w in shorter if w in longer_set)
        overlap = common / len(shorter)

        age_ms = (time.monotonic() - newest.timestamp) * 1000
        return overlap >= settings.turn_partial_stability_ratio and age_ms >= settings.turn_stable_wait_ms
