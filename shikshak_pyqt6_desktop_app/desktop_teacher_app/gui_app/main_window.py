"""
Shikshak Desktop — dark orb-based voice tutor UI.

Connects to the same LiveSessionWorker; only the visual layer changed.
"""
from __future__ import annotations

import math
import random
from datetime import datetime
from typing import Optional

from PyQt6.QtCore import (
    QEasingCurve,
    QPointF,
    QPropertyAnimation,
    QRectF,
    QThread,
    QTimer,
    Qt,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QFont,
    QFontDatabase,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui_app.workers.live_session_worker import LiveSessionWorker
from gui_app.preloader import start_preload_thread, get_preloaded

# ── Palette ────────────────────────────────────────────────────
BG          = "#0a0a0f"
BG_CARD     = "#0f0f16"
BG_PANEL    = "#0c0c13"
BORDER      = "#1a1a24"
CYAN        = "#5de4d4"
ORANGE      = "#ff7849"
VIOLET      = "#b49aff"
TEXT        = "#d4d2cf"
TEXT_DIM    = "#4a4954"
TEXT_GHOST  = "#14141c"


# ═══════════════════════════════════════════════════════════════
# Animated Orb Widget
# ═══════════════════════════════════════════════════════════════
class OrbWidget(QWidget):
    """Animated pulsing orb that changes color/glow by state."""

    clicked = pyqtSignal()

    # States: idle, loading, listening, thinking, speaking, stopped
    _STATE_COLORS = {
        "idle":      (QColor(90, 90, 110),  QColor(30, 30, 42)),
        "loading":   (QColor(180, 154, 255), QColor(30, 24, 52)),
        "listening": (QColor(93, 228, 212),  QColor(18, 36, 42)),
        "thinking":  (QColor(180, 154, 255), QColor(32, 24, 52)),
        "speaking":  (QColor(255, 120, 73),  QColor(42, 24, 18)),
        "stopped":   (QColor(90, 90, 110),  QColor(30, 30, 42)),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(220, 220)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._state = "idle"
        self._pulse = 0.0
        self._ring_angle = 0.0
        self._hover = False

        # Pulse animation
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(25)

    def set_state(self, state: str):
        s = state.lower()
        if s in ("starting", "loading"):
            s = "loading"
        elif s in ("error",):
            s = "stopped"
        elif s not in self._STATE_COLORS:
            s = "idle"
        self._state = s
        self.update()

    def _tick(self):
        self._pulse += 0.045
        speed_map = {"idle": 0.3, "loading": 1.2, "listening": 0.8, "thinking": 1.5, "speaking": 1.0, "stopped": 0.2}
        self._ring_angle += speed_map.get(self._state, 0.5)
        self.update()

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2
        accent, bg_tint = self._STATE_COLORS.get(self._state, self._STATE_COLORS["idle"])

        pulse_val = (math.sin(self._pulse) + 1) / 2  # 0..1

        # ── Outer glow ──
        glow_r = 95 + pulse_val * 8
        glow_color = QColor(accent)
        glow_color.setAlpha(int(18 + pulse_val * 14))
        grad = QRadialGradient(cx, cy, glow_r)
        grad.setColorAt(0, glow_color)
        grad.setColorAt(1, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), glow_r, glow_r)

        # ── Orbit rings ──
        for i, (radius, alpha_base, direction) in enumerate([
            (82, 0.12, 1), (92, 0.07, -1), (102, 0.04, 0.6)
        ]):
            ring_color = QColor(accent)
            ring_alpha = alpha_base + pulse_val * 0.06
            ring_color.setAlpha(int(ring_alpha * 255))
            pen = QPen(ring_color, 1.0)
            pen.setStyle(Qt.PenStyle.SolidLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)

            # Draw partial arc
            angle = self._ring_angle * direction * (0.8 + i * 0.3)
            start = int(angle * 16) % (360 * 16)
            span = int(120 * 16)  # 120 degree arc
            rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
            p.drawArc(rect, start, span)

        # ── Main orb ──
        orb_r = 58 + (2 if self._hover else 0)
        # Body gradient
        orb_grad = QRadialGradient(cx - 12, cy - 12, orb_r * 1.3)
        orb_grad.setColorAt(0, bg_tint)
        orb_grad.setColorAt(1, QColor(10, 10, 16))
        p.setBrush(QBrush(orb_grad))

        # Subtle border
        border_color = QColor(accent)
        border_color.setAlpha(int(25 + pulse_val * 20))
        p.setPen(QPen(border_color, 1.2))
        p.drawEllipse(QPointF(cx, cy), orb_r, orb_r)

        # ── Inner highlight ──
        hi = QRadialGradient(cx - 15, cy - 18, orb_r * 0.6)
        hi.setColorAt(0, QColor(255, 255, 255, 8))
        hi.setColorAt(1, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(hi))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), orb_r - 2, orb_r - 2)

        # ── Pulse ring (active states) ──
        if self._state in ("listening", "thinking", "speaking"):
            pulse_r = orb_r + 6 + pulse_val * 14
            pulse_alpha = int((1 - pulse_val) * 40)
            ring_c = QColor(accent)
            ring_c.setAlpha(pulse_alpha)
            p.setPen(QPen(ring_c, 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, cy), pulse_r, pulse_r)

        # ── Center icon (mic or stop) ──
        icon_color = QColor(accent)
        icon_color.setAlpha(180 + int(pulse_val * 50))
        p.setPen(QPen(icon_color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))

        if self._state == "stopped":
            # Stop square
            s = 12
            p.setBrush(QBrush(icon_color))
            p.drawRoundedRect(QRectF(cx - s, cy - s, s * 2, s * 2), 3, 3)
        else:
            # Mic icon
            p.setBrush(Qt.BrushStyle.NoBrush)
            mic_w, mic_h = 7, 11
            p.drawRoundedRect(QRectF(cx - mic_w, cy - mic_h - 2, mic_w * 2, mic_h * 2), mic_w, mic_w)
            # Mic base arc
            arc_rect = QRectF(cx - 14, cy - 12, 28, 28)
            p.drawArc(arc_rect, 210 * 16, 120 * 16)
            # Stem
            p.drawLine(QPointF(cx, cy + 14), QPointF(cx, cy + 20))
            p.drawLine(QPointF(cx - 6, cy + 20), QPointF(cx + 6, cy + 20))

        p.end()


# ═══════════════════════════════════════════════════════════════
# Transcript Bubble
# ═══════════════════════════════════════════════════════════════
class TranscriptBubble(QFrame):
    def __init__(self, text: str, role: str, parent=None):
        super().__init__(parent)
        is_student = role == "user"
        accent = CYAN if is_student else ORANGE
        bg = "rgba(93,228,212,0.05)" if is_student else "rgba(255,120,73,0.05)"
        border = "rgba(93,228,212,0.12)" if is_student else "rgba(255,120,73,0.12)"
        align = "right" if is_student else "left"
        prefix = "You" if is_student else "Teacher"
        ts = datetime.now().strftime("%H:%M")

        self.setStyleSheet(f"""
            TranscriptBubble {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 14px;
                padding: 10px 16px;
                margin: 2px {'0px 2px 24px' if is_student else '24px 2px 0px'};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        header = QLabel(f"{prefix}  ·  {ts}")
        header.setStyleSheet(f"color: {accent}; font-size: 10px; font-weight: 600; letter-spacing: 1px; border: none; background: transparent; padding: 0;")
        header.setAlignment(Qt.AlignmentFlag.AlignLeft if not is_student else Qt.AlignmentFlag.AlignRight)

        body = QLabel(text)
        body.setWordWrap(True)
        body.setStyleSheet(f"color: {TEXT}; font-size: 13px; line-height: 1.5; border: none; background: transparent; padding: 0;")
        body.setAlignment(Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(header)
        layout.addWidget(body)


# ═══════════════════════════════════════════════════════════════
# Chat History Panel
# ═══════════════════════════════════════════════════════════════
class ChatHistory(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet(f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background: {BG_PANEL};
                width: 5px;
                border-radius: 2px;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER};
                border-radius: 2px;
                min-height: 30px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(6)
        self._layout.addStretch()
        self.setWidget(self._container)

    def add_message(self, role: str, text: str):
        if not text.strip():
            return
        bubble = TranscriptBubble(text, role)
        # Insert before the stretch
        self._layout.insertWidget(self._layout.count() - 1, bubble)
        QTimer.singleShot(50, lambda: self.verticalScrollBar().setValue(self.verticalScrollBar().maximum()))


# ═══════════════════════════════════════════════════════════════
# Status Bar (bottom)
# ═══════════════════════════════════════════════════════════════
class StatusStrip(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setStyleSheet(f"""
            StatusStrip {{
                background: {BG_PANEL};
                border-top: 1px solid {BORDER};
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(20)

        style = f"color: {TEXT_DIM}; font-size: 10px; letter-spacing: 1px; background: transparent; border: none;"

        self.tts_label = QLabel("TTS: —")
        self.tts_label.setStyleSheet(style)
        self.sv_label = QLabel("SV: —")
        self.sv_label.setStyleSheet(style)
        self.emotion_label = QLabel("EMOTION: —")
        self.emotion_label.setStyleSheet(style)
        self.note_label = QLabel("")
        self.note_label.setStyleSheet(style)
        self.note_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.note_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        for w in (self.tts_label, self.sv_label, self.emotion_label, self.note_label):
            layout.addWidget(w)

    def set_tts(self, text: str):
        short = "enabled" if "enabled" in text.lower() else "off"
        color = CYAN if short == "enabled" else TEXT_DIM
        self.tts_label.setText(f"TTS: {short}")
        self.tts_label.setStyleSheet(f"color: {color}; font-size: 10px; letter-spacing: 1px; background: transparent; border: none;")

    def set_sv(self, text: str):
        short = "on" if "enabled" in text.lower() else "off"
        color = VIOLET if short == "on" else TEXT_DIM
        self.sv_label.setText(f"SV: {short}")
        self.sv_label.setStyleSheet(f"color: {color}; font-size: 10px; letter-spacing: 1px; background: transparent; border: none;")

    def set_emotion(self, text: str):
        self.emotion_label.setText(f"EMOTION: {text[:40]}")

    def set_note(self, text: str):
        self.note_label.setText(text[:80])


# ═══════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    _preload_progress = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shikshak — Voice Tutor")
        self.resize(900, 700)
        self.setMinimumSize(600, 500)

        # Dark background
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {BG};
            }}
        """)

        self.worker_thread: QThread | None = None
        self.worker: LiveSessionWorker | None = None
        self._session_active = False

        self._build_ui()
        self._preload_progress.connect(self._on_preload_progress)

        # Kick off model preloading
        self._set_state_display("loading")
        self.state_label.setText("LOADING MODELS")
        start_preload_thread(on_progress=self._emit_preload_progress)

    # ── Build UI ───────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar ──
        topbar = QFrame()
        topbar.setFixedHeight(52)
        topbar.setStyleSheet(f"""
            QFrame {{
                background: {BG_PANEL};
                border-bottom: 1px solid {BORDER};
            }}
        """)
        tb_layout = QHBoxLayout(topbar)
        tb_layout.setContentsMargins(20, 0, 20, 0)

        title = QLabel("Shikshak")
        title.setStyleSheet(f"color: {TEXT}; font-size: 20px; font-weight: 300; letter-spacing: 2px; background: transparent; border: none;")

        subtitle = QLabel("voice tutor")
        subtitle.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px; font-style: italic; margin-left: 8px; background: transparent; border: none;")

        self.conn_dot = QLabel("●")
        self.conn_dot.setStyleSheet(f"color: {TEXT_DIM}; font-size: 8px; background: transparent; border: none;")
        self.conn_label = QLabel("OFFLINE")
        self.conn_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; letter-spacing: 1.5px; background: transparent; border: none;")

        tb_layout.addWidget(title)
        tb_layout.addWidget(subtitle)
        tb_layout.addStretch()
        tb_layout.addWidget(self.conn_dot)
        tb_layout.addWidget(self.conn_label)
        root.addWidget(topbar)

        # ── Body: orb centered on top, chat below ──
        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # ── Orb area (centered, fixed height) ──
        orb_area = QWidget()
        orb_area.setFixedHeight(300)
        orb_area.setStyleSheet(f"background: {BG}; border-bottom: 1px solid {BORDER};")
        orb_layout = QVBoxLayout(orb_area)
        orb_layout.setContentsMargins(0, 0, 0, 0)
        orb_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        orb_layout.addStretch(2)

        self.orb = OrbWidget()
        self.orb.clicked.connect(self._on_orb_click)
        orb_layout.addWidget(self.orb, alignment=Qt.AlignmentFlag.AlignCenter)

        self.state_label = QLabel("TAP TO BEGIN")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; font-weight: 500; letter-spacing: 3px; background: transparent; border: none;")
        orb_layout.addWidget(self.state_label)

        # Live transcription (ephemeral, below orb)
        live_row = QHBoxLayout()
        live_row.setContentsMargins(40, 4, 40, 0)
        live_row.setSpacing(16)

        self.live_student_label = QLabel("")
        self.live_student_label.setWordWrap(True)
        self.live_student_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_student_label.setStyleSheet(f"color: {CYAN}; font-size: 12px; padding: 4px 12px; background: transparent; border: none;")

        self.live_teacher_label = QLabel("")
        self.live_teacher_label.setWordWrap(True)
        self.live_teacher_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_teacher_label.setStyleSheet(f"color: {ORANGE}; font-size: 12px; padding: 4px 12px; background: transparent; border: none;")

        live_row.addWidget(self.live_student_label, stretch=1)
        live_row.addWidget(self.live_teacher_label, stretch=1)
        orb_layout.addLayout(live_row)

        orb_layout.addStretch(1)

        body.addWidget(orb_area)

        # ── Chat history (fills remaining space) ──
        chat_panel = QWidget()
        chat_panel.setStyleSheet(f"background: {BG};")
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(0, 8, 0, 0)
        chat_layout.setSpacing(0)

        chat_header = QLabel("  CONVERSATION")
        chat_header.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; letter-spacing: 2px; font-weight: 600; padding: 4px 16px; background: transparent; border: none;")
        chat_layout.addWidget(chat_header)

        self.chat_history = ChatHistory()
        chat_layout.addWidget(self.chat_history, stretch=1)

        body.addWidget(chat_panel, stretch=1)

        root.addLayout(body, stretch=1)

        # ── Bottom status strip ──
        self.status_strip = StatusStrip()
        root.addWidget(self.status_strip)

    # ── State display helpers ──────────────────────────────────
    def _set_state_display(self, state: str):
        self.orb.set_state(state)
        state_labels = {
            "idle": "TAP TO BEGIN",
            "loading": "LOADING MODELS",
            "listening": "LISTENING",
            "thinking": "THINKING",
            "speaking": "SPEAKING",
            "interrupted": "INTERRUPTED",
            "starting": "STARTING",
            "stopped": "SESSION ENDED",
        }
        label = state_labels.get(state.lower(), state.upper())
        self.state_label.setText(label)

        # Color the label
        color_map = {
            "listening": CYAN, "thinking": VIOLET, "speaking": ORANGE,
            "interrupted": ORANGE,
        }
        c = color_map.get(state.lower(), TEXT_DIM)
        self.state_label.setStyleSheet(f"color: {c}; font-size: 11px; font-weight: 500; letter-spacing: 3px; background: transparent; border: none;")

    def _set_connection(self, ok: bool):
        if ok:
            self.conn_dot.setStyleSheet(f"color: {CYAN}; font-size: 8px; background: transparent; border: none;")
            self.conn_label.setText("LIVE")
            self.conn_label.setStyleSheet(f"color: {CYAN}; font-size: 10px; letter-spacing: 1.5px; background: transparent; border: none; opacity: 0.7;")
        else:
            self.conn_dot.setStyleSheet(f"color: {TEXT_DIM}; font-size: 8px; background: transparent; border: none;")
            self.conn_label.setText("OFFLINE")
            self.conn_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; letter-spacing: 1.5px; background: transparent; border: none;")

    # ── Preload callbacks ──────────────────────────────────────
    def _emit_preload_progress(self, msg: str):
        self._preload_progress.emit(msg)

    def _on_preload_progress(self, msg: str):
        self.status_strip.set_note(msg)
        models = get_preloaded()
        if models.ready:
            self._set_state_display("idle")
            self.status_strip.set_tts(models.tts_status)
            self.status_strip.set_sv(models.sv_status)
            self.status_strip.set_note("Ready — click the orb to start")
        elif models.error:
            self._set_state_display("stopped")
            self.status_strip.set_note(f"Preload failed: {models.error}")

    # ── Orb click → start/stop ─────────────────────────────────
    def _on_orb_click(self):
        if self._session_active:
            self.stop_session()
        else:
            self.start_session()

    def start_session(self):
        models = get_preloaded()
        if not models.ready and not models.error:
            return  # still loading
        if self.worker_thread is not None:
            return

        self.worker_thread = QThread()
        self.worker = LiveSessionWorker()
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self._on_finished)

        # Connect all signals
        self.worker.status_changed.connect(self._on_status_changed)
        self.worker.note_changed.connect(self.status_strip.set_note)
        self.worker.live_student_text.connect(self._on_live_student)
        self.worker.final_student_text.connect(lambda t: self.chat_history.add_message("user", t))
        self.worker.live_teacher_text.connect(self._on_live_teacher)
        self.worker.final_teacher_text.connect(lambda t: self.chat_history.add_message("assistant", t))
        self.worker.emotion_changed.connect(self.status_strip.set_emotion)
        self.worker.session_ready.connect(self._on_session_ready)
        self.worker.health_report.connect(self._on_health_report)
        self.worker.error_occurred.connect(self._on_error)

        self._session_active = True
        self._set_connection(True)
        self.worker_thread.start()

    def stop_session(self):
        if self.worker is not None:
            self.status_strip.set_note("Stopping session…")
            self.worker.stop()
        self._session_active = False
        self._set_connection(False)

    def _on_finished(self):
        self._session_active = False
        self._set_state_display("idle")
        self._set_connection(False)
        self.live_student_label.setText("")
        self.live_teacher_label.setText("")
        if self.worker_thread is not None:
            self.worker_thread.wait(2000)
        self.worker_thread = None
        self.worker = None
        self.status_strip.set_note("Session ended")

    # ── Signal handlers ────────────────────────────────────────
    def _on_status_changed(self, status: str):
        self._set_state_display(status)

    def _on_live_student(self, text: str):
        self.live_student_label.setText(text[:120] if text else "")

    def _on_live_teacher(self, text: str):
        display = text[:100] + "…" if len(text) > 100 else text
        self.live_teacher_label.setText(display if text else "")

    def _on_session_ready(self, session_id: str):
        self.status_strip.set_note(f"Session: {session_id}")

    def _on_health_report(self, kind: str, value: str):
        if kind == "TTS":
            self.status_strip.set_tts(value)
        elif kind == "Speaker verification":
            self.status_strip.set_sv(value)
        else:
            self.status_strip.set_note(f"{kind}: {value}")

    def _on_error(self, message: str):
        QMessageBox.critical(self, "Session Error", message)
        self.status_strip.set_note(message)
        self.stop_session()

    def closeEvent(self, event):
        self.stop_session()
        super().closeEvent(event)