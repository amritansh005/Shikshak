from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget


class ChatPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.live_student = QLabel("You: —")
        self.live_teacher = QLabel("Teacher: —")
        for label in (self.live_student, self.live_teacher):
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            label.setStyleSheet("font-size: 14px; padding: 8px; border-radius: 8px; background: #1e1e1e;")

        self.history = QTextEdit()
        self.history.setReadOnly(True)
        self.history.setStyleSheet("font-size: 14px;")

        layout.addWidget(self.live_student)
        layout.addWidget(self.live_teacher)
        layout.addWidget(self.history, stretch=1)

    def set_live_student(self, text: str) -> None:
        self.live_student.setText(f"You: {text or '—'}")

    def set_live_teacher(self, text: str) -> None:
        self.live_teacher.setText(f"Teacher: {text or '—'}")

    def append_message(self, role: str, text: str) -> None:
        if not text:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        prefix = "You" if role == "user" else "Teacher"
        self.history.append(f"<b>[{ts}] {prefix}:</b><br>{text}<br>")
        self.history.moveCursor(QTextCursor.MoveOperation.End)
        self.history.ensureCursorVisible()
