from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)

EMOTION_WINDOW_SIZE = 8
SMOOTHING_ALPHA = 0.40

TOPIC_COUNT_DECAY = 0.90
RECOVERY_TTL_TURNS = 2
BASELINE_MIN_TURNS = 4

ALL_TEACHING_STATES = {
    "neutral",
    "frustrated",
    "confused",
    "anxious",
    "discouraged",
    "uncertain",
    "bored",
    "confident",
    "curious",
    "engaged",
}

NEGATIVE_STATES = {"frustrated", "confused", "anxious", "discouraged", "uncertain"}
POSITIVE_STATES = {"confident", "curious", "engaged"}

SER_TO_TEACHING: Dict[str, str] = {
    "angry": "frustrated",
    "disgusted": "frustrated",
    "fearful": "anxious",
    "happy": "engaged",
    "neutral": "neutral",
    "other": "neutral",
    "sad": "discouraged",
    "surprised": "curious",
    "unknown": "neutral",
}

TEXT_TO_TEACHING: Dict[str, str] = {
    "frustrated": "frustrated",
    "confused": "confused",
    "bored": "bored",
    "confident": "confident",
    "curious": "curious",
    "anxious": "anxious",
    "neutral": "neutral",
}

BASE_SIGNAL_WEIGHTS = {
    "audio": 0.45,
    "text": 0.35,
    "prosody": 0.20,
}

STATE_INTENSITY = {
    "frustrated": 1.00,
    "confused": 0.80,
    "anxious": 0.70,
    "discouraged": 0.75,
    "uncertain": 0.50,
    "bored": 0.35,
    "neutral": 0.00,
    "curious": -0.30,
    "engaged": -0.40,
    "confident": -0.60,
}


@dataclass
class RawEmotionSignals:
    text_label: str = "neutral"
    text_confidence: float = 0.5

    audio_label: str = "neutral"
    audio_confidence: float = 0.5
    audio_all_scores: Dict[str, float] = field(default_factory=dict)

    speech_rate_sps: float = 0.0
    pause_ratio: float = 0.0
    filled_pauses: int = 0
    pitch_mean_hz: float = 0.0
    pitch_std_hz: float = 0.0


@dataclass
class EmotionSnapshot:
    turn_number: int
    timestamp: float
    raw: RawEmotionSignals

    topic: str = ""

    fused_distribution: Dict[str, float] = field(default_factory=dict)
    smoothed_distribution: Dict[str, float] = field(default_factory=dict)

    teaching_state: str = "neutral"
    teaching_confidence: float = 0.5
    secondary_state: str = "neutral"
    secondary_confidence: float = 0.0

    smoothed_state: str = "neutral"
    smoothed_confidence: float = 0.5
    smoothed_secondary_state: str = "neutral"
    smoothed_secondary_confidence: float = 0.0

    negative_pressure: float = 0.0


@dataclass
class TopicEmotionMemory:
    topic: str
    turn_count: float = 0.0
    negative_turns: float = 0.0
    confused_turns: float = 0.0
    frustrated_turns: float = 0.0
    peak_negative_streak: int = 0
    current_negative_streak: int = 0
    last_state: str = "neutral"
    last_recovery_turn: Optional[int] = None


@dataclass
class EmotionTrend:
    current_state: str = "neutral"
    current_confidence: float = 0.5
    secondary_state: str = "neutral"
    secondary_confidence: float = 0.0

    smoothed_state: str = "neutral"
    smoothed_confidence: float = 0.5
    smoothed_secondary_state: str = "neutral"
    smoothed_secondary_confidence: float = 0.0

    trend: str = "stable"
    turns_in_current_state: int = 1
    dominant_state: str = "neutral"
    topic_memory: Optional[TopicEmotionMemory] = None
    negative_pressure: float = 0.0


@dataclass
class TeachingDirective:
    raw_text_label: str = "neutral"
    raw_audio_label: str = "neutral"

    teaching_state: str = "neutral"
    teaching_confidence: float = 0.5
    smoothed_state: str = "neutral"
    smoothed_confidence: float = 0.5

    secondary_state: str = "neutral"
    secondary_confidence: float = 0.0
    smoothed_secondary_state: str = "neutral"
    smoothed_secondary_confidence: float = 0.0

    trend: str = "stable"
    instruction: str = ""


@dataclass
class SpeakerProsodyBaseline:
    speech_rate_values: List[float] = field(default_factory=list)
    pause_ratio_values: List[float] = field(default_factory=list)
    pitch_std_values: List[float] = field(default_factory=list)

    def update(self, raw: RawEmotionSignals) -> None:
        if raw.speech_rate_sps > 0:
            self.speech_rate_values.append(raw.speech_rate_sps)
            self.speech_rate_values = self.speech_rate_values[-20:]

        if raw.pause_ratio > 0:
            self.pause_ratio_values.append(raw.pause_ratio)
            self.pause_ratio_values = self.pause_ratio_values[-20:]

        if raw.pitch_std_hz > 0:
            self.pitch_std_values.append(raw.pitch_std_hz)
            self.pitch_std_values = self.pitch_std_values[-20:]

    def ready(self) -> bool:
        return (
            len(self.speech_rate_values) >= BASELINE_MIN_TURNS
            and len(self.pause_ratio_values) >= BASELINE_MIN_TURNS
        )

    @staticmethod
    def _avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def speech_rate_avg(self) -> float:
        return self._avg(self.speech_rate_values)

    @property
    def pause_ratio_avg(self) -> float:
        return self._avg(self.pause_ratio_values)

    @property
    def pitch_std_avg(self) -> float:
        return self._avg(self.pitch_std_values)


def _empty_distribution() -> Dict[str, float]:
    return {state: 0.0 for state in ALL_TEACHING_STATES}


def _normalize_distribution(dist: Dict[str, float]) -> Dict[str, float]:
    clean = {k: max(0.0, float(v)) for k, v in dist.items() if k in ALL_TEACHING_STATES}
    total = sum(clean.values())
    if total <= 0:
        return {state: (1.0 if state == "neutral" else 0.0) for state in ALL_TEACHING_STATES}
    return {state: clean.get(state, 0.0) / total for state in ALL_TEACHING_STATES}


def _top_two_states(dist: Dict[str, float]) -> Tuple[Tuple[str, float], Tuple[str, float]]:
    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    if not items:
        return ("neutral", 1.0), ("neutral", 0.0)
    if len(items) == 1:
        return items[0], ("neutral", 0.0)
    return items[0], items[1]


def _compute_negative_pressure(dist: Dict[str, float]) -> float:
    score = 0.0
    for state, prob in dist.items():
        score += prob * STATE_INTENSITY.get(state, 0.0)
    return round(score, 4)


def _confidence_from_distribution(dist: Dict[str, float]) -> Tuple[float, float]:
    (best_state, best_prob), (_, second_prob) = _top_two_states(dist)
    dominance = max(0.0, best_prob - second_prob)
    absolute = best_prob
    combined = (0.6 * dominance) + (0.4 * absolute)
    return round(combined, 3), round(absolute, 3)


def _signal_reliabilities(raw: RawEmotionSignals) -> Dict[str, float]:
    reliabilities = {"audio": 1.0, "text": 1.0, "prosody": 1.0}

    if raw.audio_label == "unknown":
        reliabilities["audio"] *= 0.35
    if raw.audio_confidence < 0.45:
        reliabilities["audio"] *= 0.6
    if raw.audio_all_scores:
        top_ser = max(raw.audio_all_scores.values())
        if top_ser < 0.45:
            reliabilities["audio"] *= 0.65

    if raw.text_label == "neutral" and raw.text_confidence <= 0.5:
        reliabilities["text"] *= 0.7
    if raw.text_confidence < 0.45:
        reliabilities["text"] *= 0.65

    prosody_valid = (
        raw.speech_rate_sps > 0
        or raw.pause_ratio > 0
        or raw.filled_pauses > 0
        or raw.pitch_std_hz > 0
    )
    if not prosody_valid:
        reliabilities["prosody"] *= 0.25

    if raw.speech_rate_sps <= 0:
        reliabilities["prosody"] *= 0.75
    if raw.pitch_std_hz <= 0:
        reliabilities["prosody"] *= 0.85

    return reliabilities


def _adaptive_weights(raw: RawEmotionSignals) -> Dict[str, float]:
    reliabilities = _signal_reliabilities(raw)

    weighted = {}
    for source, base_weight in BASE_SIGNAL_WEIGHTS.items():
        weighted[source] = base_weight * reliabilities[source]

    total = sum(weighted.values())
    if total <= 0:
        return BASE_SIGNAL_WEIGHTS.copy()

    return {k: v / total for k, v in weighted.items()}


def _prosody_to_distribution(
    raw: RawEmotionSignals,
    baseline: Optional[SpeakerProsodyBaseline],
) -> Dict[str, float]:
    dist = _empty_distribution()

    if (
        raw.speech_rate_sps <= 0
        and raw.pause_ratio <= 0
        and raw.filled_pauses <= 0
        and raw.pitch_std_hz <= 0
    ):
        dist["neutral"] = 1.0
        return dist

    if baseline and baseline.ready():
        rate_avg = baseline.speech_rate_avg or raw.speech_rate_sps
        pause_avg = baseline.pause_ratio_avg or raw.pause_ratio
        pitch_std_avg = baseline.pitch_std_avg or raw.pitch_std_hz

        slow_relative = raw.speech_rate_sps < (0.72 * rate_avg) if rate_avg > 0 else False
        fast_relative = raw.speech_rate_sps > (1.28 * rate_avg) if rate_avg > 0 else False
        pausy_relative = raw.pause_ratio > (1.35 * pause_avg) if pause_avg > 0 else False
        expressive_relative = raw.pitch_std_hz > (1.30 * pitch_std_avg) if pitch_std_avg > 0 else False
    else:
        slow_relative = raw.speech_rate_sps > 0 and raw.speech_rate_sps < 2.5
        fast_relative = raw.speech_rate_sps > 5.5
        pausy_relative = raw.pause_ratio > 0.45
        expressive_relative = raw.pitch_std_hz > 60 and raw.speech_rate_sps > 3.0

    if slow_relative and pausy_relative:
        dist["uncertain"] += 0.65
        dist["discouraged"] += 0.15

    if raw.filled_pauses >= 3:
        dist["uncertain"] += 0.45
        dist["anxious"] += 0.20

    if fast_relative:
        dist["anxious"] += 0.50
        dist["engaged"] += 0.15

    if expressive_relative:
        dist["engaged"] += 0.35
        dist["curious"] += 0.20

    if sum(dist.values()) == 0:
        dist["neutral"] = 1.0

    return _normalize_distribution(dist)


def _text_to_distribution(raw: RawEmotionSignals) -> Dict[str, float]:
    dist = _empty_distribution()
    teaching = TEXT_TO_TEACHING.get(raw.text_label, "neutral")
    conf = max(0.0, min(1.0, raw.text_confidence))

    dist[teaching] += conf
    dist["neutral"] += max(0.0, 1.0 - conf)

    return _normalize_distribution(dist)


def _audio_to_distribution(raw: RawEmotionSignals) -> Dict[str, float]:
    dist = _empty_distribution()

    if raw.audio_all_scores:
        for ser_label, score in raw.audio_all_scores.items():
            teaching = SER_TO_TEACHING.get(ser_label, "neutral")
            dist[teaching] += max(0.0, float(score))
    else:
        teaching = SER_TO_TEACHING.get(raw.audio_label, "neutral")
        conf = max(0.0, min(1.0, raw.audio_confidence))
        dist[teaching] += conf
        dist["neutral"] += max(0.0, 1.0 - conf)

    return _normalize_distribution(dist)


def _fuse_signals(
    raw: RawEmotionSignals,
    baseline: Optional[SpeakerProsodyBaseline],
) -> Dict[str, float]:
    weights = _adaptive_weights(raw)

    text_dist = _text_to_distribution(raw)
    audio_dist = _audio_to_distribution(raw)
    prosody_dist = _prosody_to_distribution(raw, baseline)

    fused = _empty_distribution()
    for state in ALL_TEACHING_STATES:
        fused[state] = (
            (weights["text"] * text_dist.get(state, 0.0))
            + (weights["audio"] * audio_dist.get(state, 0.0))
            + (weights["prosody"] * prosody_dist.get(state, 0.0))
        )

    fused = _normalize_distribution(fused)

    logger.debug(
        "Fusion | weights=%s | text_dist=%s | audio_dist=%s | prosody_dist=%s | fused=%s",
        weights,
        text_dist,
        audio_dist,
        prosody_dist,
        fused,
    )

    return fused


def _smooth_distribution(
    current_dist: Dict[str, float],
    window: Deque[EmotionSnapshot],
) -> Dict[str, float]:
    if not window:
        return current_dist

    smoothed = _empty_distribution()

    history = list(window)
    for i, snap in enumerate(history):
        age = len(history) - i
        weight = (1 - SMOOTHING_ALPHA) ** age
        # Use fused_distribution (raw per-turn signal), NOT smoothed_distribution,
        # to prevent double-smoothing compounding where old data accumulates
        # more inertia than the alpha value intends.
        for state, prob in snap.fused_distribution.items():
            smoothed[state] += weight * prob

    for state, prob in current_dist.items():
        smoothed[state] += SMOOTHING_ALPHA * prob

    return _normalize_distribution(smoothed)


class EmotionStateService:
    def __init__(self) -> None:
        self._windows: Dict[str, Deque[EmotionSnapshot]] = {}
        self._turn_counters: Dict[str, int] = {}
        self._topic_memories: Dict[str, Dict[str, TopicEmotionMemory]] = {}
        self._speaker_baselines: Dict[str, SpeakerProsodyBaseline] = {}

    def _get_window(self, session_id: str) -> Deque[EmotionSnapshot]:
        if session_id not in self._windows:
            self._windows[session_id] = deque(maxlen=EMOTION_WINDOW_SIZE)
            self._turn_counters[session_id] = 0
            self._topic_memories[session_id] = {}
            self._speaker_baselines[session_id] = SpeakerProsodyBaseline()
        return self._windows[session_id]

    def record_turn(
        self,
        session_id: str,
        emotion_data: Dict,
        topic: str = "",
    ) -> TeachingDirective:
        window = self._get_window(session_id)
        self._turn_counters[session_id] += 1
        turn_num = self._turn_counters[session_id]

        text_emotion = emotion_data.get("text_emotion", {})
        audio_emotion = emotion_data.get("audio_emotion", {})
        prosody = emotion_data.get("prosody", {})

        raw = RawEmotionSignals(
            text_label=text_emotion.get("label", "neutral"),
            text_confidence=text_emotion.get("confidence", 0.5),
            audio_label=audio_emotion.get("label", "neutral"),
            audio_confidence=audio_emotion.get("confidence", 0.5),
            audio_all_scores=audio_emotion.get("all_scores", {}) or {},
            speech_rate_sps=prosody.get("speech_rate_sps", 0.0),
            pause_ratio=prosody.get("pause_ratio", 0.0),
            filled_pauses=prosody.get("filled_pauses", 0),
            pitch_mean_hz=prosody.get("pitch_mean_hz", 0.0),
            pitch_std_hz=prosody.get("pitch_std_hz", 0.0),
        )

        baseline = self._speaker_baselines[session_id]
        fused_distribution = _fuse_signals(raw, baseline)
        smoothed_distribution = _smooth_distribution(fused_distribution, window)

        (primary_state, primary_prob), (secondary_state, secondary_prob) = _top_two_states(fused_distribution)
        (smoothed_primary, smoothed_primary_prob), (smoothed_secondary, smoothed_secondary_prob) = _top_two_states(smoothed_distribution)

        teaching_confidence, _ = _confidence_from_distribution(fused_distribution)
        smoothed_confidence, _ = _confidence_from_distribution(smoothed_distribution)

        snapshot = EmotionSnapshot(
            turn_number=turn_num,
            timestamp=time.time(),
            raw=raw,
            topic=topic,
            fused_distribution=fused_distribution,
            smoothed_distribution=smoothed_distribution,
            teaching_state=primary_state,
            teaching_confidence=teaching_confidence,
            secondary_state=secondary_state,
            secondary_confidence=round(secondary_prob, 3),
            smoothed_state=smoothed_primary,
            smoothed_confidence=smoothed_confidence,
            smoothed_secondary_state=smoothed_secondary,
            smoothed_secondary_confidence=round(smoothed_secondary_prob, 3),
            negative_pressure=_compute_negative_pressure(smoothed_distribution),
        )

        window.append(snapshot)
        baseline.update(raw)

        topic_mem = self._update_topic_memory(session_id, snapshot)
        trend = self._compute_trend(window, topic_mem)
        directive = self._apply_policy(trend, snapshot)

        logger.info(
            "Emotion state | session=%s | turn=%d | text=%s | audio=%s | fused=%s (%.2f) | smoothed=%s (%.2f) | trend=%s | topic=%s",
            session_id,
            turn_num,
            raw.text_label,
            raw.audio_label,
            snapshot.teaching_state,
            snapshot.teaching_confidence,
            snapshot.smoothed_state,
            snapshot.smoothed_confidence,
            trend.trend,
            topic or "(none)",
        )

        logger.debug(
            "Emotion detail | fused=%s | smoothed=%s | neg_pressure=%.3f | secondary=%s/%.3f",
            snapshot.fused_distribution,
            snapshot.smoothed_distribution,
            snapshot.negative_pressure,
            snapshot.smoothed_secondary_state,
            snapshot.smoothed_secondary_confidence,
        )

        return directive

    def get_directive_for_text_only(
        self,
        session_id: str,
        text_emotion_label: str = "neutral",
        text_emotion_confidence: float = 0.5,
    ) -> TeachingDirective:
        emotion_data = {
            "text_emotion": {
                "label": text_emotion_label,
                "confidence": text_emotion_confidence,
            },
            "audio_emotion": {
                "label": "neutral",
                "confidence": 0.5,
            },
            "prosody": {},
        }
        return self.record_turn(session_id, emotion_data)

    def _update_topic_memory(
        self,
        session_id: str,
        snapshot: EmotionSnapshot,
    ) -> Optional[TopicEmotionMemory]:
        topic = snapshot.topic.strip().lower()
        if not topic:
            return None

        memories = self._topic_memories.get(session_id, {})
        if topic not in memories:
            memories[topic] = TopicEmotionMemory(topic=topic)
        self._topic_memories[session_id] = memories

        mem = memories[topic]

        mem.turn_count *= TOPIC_COUNT_DECAY
        mem.negative_turns *= TOPIC_COUNT_DECAY
        mem.confused_turns *= TOPIC_COUNT_DECAY
        mem.frustrated_turns *= TOPIC_COUNT_DECAY

        mem.turn_count += 1.0
        state = snapshot.smoothed_state

        is_negative = state in NEGATIVE_STATES
        is_positive = state in POSITIVE_STATES
        was_negative = mem.last_state in NEGATIVE_STATES

        if is_negative:
            mem.negative_turns += 1.0
            mem.current_negative_streak += 1
            mem.peak_negative_streak = max(mem.peak_negative_streak, mem.current_negative_streak)
        else:
            mem.current_negative_streak = 0

        if state == "confused":
            mem.confused_turns += 1.0
        if state == "frustrated":
            mem.frustrated_turns += 1.0

        if was_negative and is_positive:
            mem.last_recovery_turn = snapshot.turn_number

        mem.last_state = state
        return mem

    def _compute_trend(
        self,
        window: Deque[EmotionSnapshot],
        topic_mem: Optional[TopicEmotionMemory],
    ) -> EmotionTrend:
        if not window:
            return EmotionTrend()

        current = window[-1]
        states = [s.smoothed_state for s in window]

        state_counts: Dict[str, int] = {}
        for s in states:
            state_counts[s] = state_counts.get(s, 0) + 1
        dominant = max(state_counts, key=state_counts.get)

        turns_in_state = 1
        for i in range(len(states) - 2, -1, -1):
            if states[i] == current.smoothed_state:
                turns_in_state += 1
            else:
                break

        recent_snaps = list(window)[-3:]
        earlier_snaps = list(window)[-6:-3] if len(window) >= 6 else list(window)[:-3]

        recent_pressure = sum(s.negative_pressure for s in recent_snaps) / len(recent_snaps) if recent_snaps else 0.0
        earlier_pressure = sum(s.negative_pressure for s in earlier_snaps) / len(earlier_snaps) if earlier_snaps else recent_pressure

        delta = recent_pressure - earlier_pressure
        trend = "stable"

        if delta > 0.18 and current.smoothed_state in NEGATIVE_STATES:
            trend = "escalating"
        elif delta < -0.18 and current.smoothed_state not in NEGATIVE_STATES:
            trend = "de-escalating"

        if topic_mem and topic_mem.last_recovery_turn is not None:
            if (current.turn_number - topic_mem.last_recovery_turn) <= RECOVERY_TTL_TURNS:
                if current.smoothed_state in POSITIVE_STATES:
                    trend = "recovering"

        return EmotionTrend(
            current_state=current.teaching_state,
            current_confidence=current.teaching_confidence,
            secondary_state=current.secondary_state,
            secondary_confidence=current.secondary_confidence,
            smoothed_state=current.smoothed_state,
            smoothed_confidence=current.smoothed_confidence,
            smoothed_secondary_state=current.smoothed_secondary_state,
            smoothed_secondary_confidence=current.smoothed_secondary_confidence,
            trend=trend,
            turns_in_current_state=turns_in_state,
            dominant_state=dominant,
            topic_memory=topic_mem,
            negative_pressure=current.negative_pressure,
        )

    def _apply_policy(
        self,
        trend: EmotionTrend,
        snapshot: EmotionSnapshot,
    ) -> TeachingDirective:
        state = trend.smoothed_state
        conf = trend.smoothed_confidence
        secondary = trend.smoothed_secondary_state
        secondary_conf = trend.smoothed_secondary_confidence
        t = trend.trend
        turns = trend.turns_in_current_state
        tmem = trend.topic_memory

        low_conf = conf < 0.22
        medium_conf = 0.22 <= conf < 0.38

        instruction = ""

        if low_conf:
            instruction = (
                "The student's emotional state is uncertain. "
                "Avoid making strong assumptions. Teach gently, ask a quick check question, "
                "and watch for whether they seem confused, discouraged, or ready to continue."
            )
        elif state == "frustrated":
            if t == "escalating" or turns >= 3:
                instruction = (
                    "The student is frustrated and it is getting worse. "
                    "Simplify your explanation significantly. Use a completely different approach or analogy. "
                    "Acknowledge their effort warmly. Keep your response short and encouraging."
                )
            elif tmem and tmem.frustrated_turns >= 2.2:
                instruction = (
                    "The student has been frustrated with this topic multiple times. "
                    "Try a fundamentally different angle. Consider asking what specific part is blocking them."
                )
            elif medium_conf:
                instruction = (
                    "The student may be frustrated. "
                    "Be extra patient, simplify slightly, and check whether a particular step is bothering them."
                )
            else:
                instruction = (
                    "The student seems frustrated. "
                    "Simplify your explanation. Be encouraging and patient. "
                    "Try a different example if the current one is not working."
                )

        elif state == "confused":
            if tmem and tmem.confused_turns >= 3.0:
                instruction = (
                    "The student has been repeatedly confused about this topic across multiple turns. "
                    "The current approach is not working. Start over from the most basic foundation. "
                    "Use a completely different example or analogy. Check each small step before moving on."
                )
            elif turns >= 2:
                instruction = (
                    "The student has been confused for multiple turns. "
                    "Break the concept into smaller pieces. Start from the most basic part. "
                    "Check understanding before moving forward."
                )
            elif medium_conf:
                instruction = (
                    "The student may be confused. "
                    "Explain a little more carefully and ask which part feels unclear."
                )
            else:
                instruction = (
                    "The student seems confused. "
                    "Explain more carefully. Use a concrete example. "
                    "Ask if a specific part is unclear."
                )

        elif state == "discouraged":
            instruction = (
                "The student seems discouraged or disheartened about this topic. "
                "Be warm and encouraging. Highlight something they already got right. "
                "Make the next step feel small and achievable."
            )

        elif state == "uncertain":
            instruction = (
                "The student sounds uncertain based on their speech pattern. "
                "Give gentle encouragement. Confirm what they already understand before adding new information."
            )

        elif state == "anxious":
            instruction = (
                "The student seems anxious about this topic. "
                "Be reassuring. Normalize the difficulty. "
                "Focus on what they already know and build from there."
            )

        elif state == "bored":
            if turns >= 2:
                instruction = (
                    "The student seems disengaged. "
                    "Pick up the pace. Offer a more challenging angle or an interesting real-world connection. "
                    "Ask if they want to move to a harder subtopic."
                )
            else:
                instruction = (
                    "The student may be finding this too easy. "
                    "Keep the pace brisk. Consider offering a challenge question."
                )

        elif state == "confident":
            if t == "recovering":
                instruction = (
                    "The student was struggling earlier but is now getting it. "
                    "Reinforce their understanding with a quick check or follow-up question. "
                    "Acknowledge their progress warmly."
                )
            else:
                instruction = (
                    "The student seems confident. "
                    "You can move at a normal or slightly faster pace. "
                    "Consider offering a deeper insight or follow-up challenge."
                )

        elif state == "engaged":
            instruction = (
                "The student seems engaged and positive. "
                "Maintain the current pace and energy. Build on this momentum."
            )

        elif state == "curious":
            instruction = (
                "The student is curious and engaged. "
                "Lean into their curiosity. Give a bit more depth or an interesting detail. "
                "Encourage their questions."
            )

        else:
            instruction = ""

        if secondary in NEGATIVE_STATES and secondary_conf >= 0.20 and state in POSITIVE_STATES:
            instruction += (
                " Also note: there may still be some lingering uncertainty or strain underneath the positive signals, "
                "so do a quick understanding check before moving too fast."
            )
        elif secondary in POSITIVE_STATES and secondary_conf >= 0.20 and state in NEGATIVE_STATES:
            instruction += (
                " The student still shows some positive engagement, so keep your tone hopeful and collaborative."
            )

        return TeachingDirective(
            raw_text_label=snapshot.raw.text_label,
            raw_audio_label=snapshot.raw.audio_label,
            teaching_state=snapshot.teaching_state,
            teaching_confidence=snapshot.teaching_confidence,
            smoothed_state=snapshot.smoothed_state,
            smoothed_confidence=snapshot.smoothed_confidence,
            secondary_state=snapshot.secondary_state,
            secondary_confidence=snapshot.secondary_confidence,
            smoothed_secondary_state=snapshot.smoothed_secondary_state,
            smoothed_secondary_confidence=snapshot.smoothed_secondary_confidence,
            trend=t,
            instruction=instruction.strip(),
        )