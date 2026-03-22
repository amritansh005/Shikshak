from __future__ import annotations

from PyQt6.QtWidgets import QFormLayout, QLabel, QWidget


class StatusPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state_value = QLabel("Idle")
        self.session_value = QLabel("—")
        self.tts_value = QLabel("Unknown")
        self.sv_value = QLabel("Unknown")
        self.emotion_value = QLabel("—")
        self.note_value = QLabel("Ready")
        self.note_value.setWordWrap(True)

        layout = QFormLayout(self)
        layout.addRow("State", self.state_value)
        layout.addRow("Session", self.session_value)
        layout.addRow("TTS", self.tts_value)
        layout.addRow("Speaker verification", self.sv_value)
        layout.addRow("Emotion", self.emotion_value)
        layout.addRow("Note", self.note_value)

    def set_state(self, text: str) -> None:
        self.state_value.setText(text)

    def set_session(self, text: str) -> None:
        self.session_value.setText(text)

    def set_tts(self, text: str) -> None:
        self.tts_value.setText(text)

    def set_sv(self, text: str) -> None:
        self.sv_value.setText(text)

    def set_emotion(self, text: str) -> None:
        self.emotion_value.setText(text)

    def set_note(self, text: str) -> None:
        self.note_value.setText(text)
