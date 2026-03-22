from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui_app.widgets.chat_panel import ChatPanel
from gui_app.widgets.status_panel import StatusPanel
from gui_app.workers.live_session_worker import LiveSessionWorker
from gui_app.preloader import start_preload_thread, get_preloaded


class MainWindow(QMainWindow):
    # Signal to receive preload progress from the background thread
    _preload_progress = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Shikshak AI Teacher")
        self.resize(1200, 800)

        self.worker_thread: QThread | None = None
        self.worker: LiveSessionWorker | None = None

        container = QWidget()
        self.setCentralWidget(container)
        root = QVBoxLayout(container)

        top = QHBoxLayout()
        self.title = QLabel("Shikshak AI Teacher — Live Desktop Client")
        self.start_btn = QPushButton("Start Session")
        self.stop_btn = QPushButton("End Session")
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(False)  # disabled until preload completes
        top.addWidget(self.title)
        top.addStretch(1)
        top.addWidget(self.start_btn)
        top.addWidget(self.stop_btn)
        root.addLayout(top)

        body = QHBoxLayout()
        self.chat_panel = ChatPanel()
        self.status_panel = StatusPanel()
        body.addWidget(self.chat_panel, stretch=3)
        body.addWidget(self.status_panel, stretch=1)
        root.addLayout(body, stretch=1)

        self.start_btn.clicked.connect(self.start_session)
        self.stop_btn.clicked.connect(self.stop_session)

        # Wire up preload progress signal
        self._preload_progress.connect(self._on_preload_progress)

        # Start preloading immediately
        self.status_panel.set_state("Loading")
        self.status_panel.set_note("Pre-loading models...")
        start_preload_thread(on_progress=self._emit_preload_progress)

    def _emit_preload_progress(self, msg: str) -> None:
        """Called from background thread — emit signal to update GUI safely."""
        self._preload_progress.emit(msg)

    def _on_preload_progress(self, msg: str) -> None:
        """Runs on main thread via signal."""
        self.status_panel.set_note(msg)
        models = get_preloaded()

        if models.ready:
            self.start_btn.setEnabled(True)
            self.status_panel.set_state("Idle")
            self.status_panel.set_tts(models.tts_status)
            self.status_panel.set_sv(models.sv_status)
            self.status_panel.set_note("Ready — click Start Session")
        elif models.error:
            self.status_panel.set_state("Error")
            self.status_panel.set_note(f"Preload failed: {models.error}")
            self.start_btn.setEnabled(True)  # allow retry

    def start_session(self) -> None:
        if self.worker_thread is not None:
            return
        self.worker_thread = QThread()
        self.worker = LiveSessionWorker()
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self._on_finished)

        self.worker.status_changed.connect(self.status_panel.set_state)
        self.worker.note_changed.connect(self.status_panel.set_note)
        self.worker.live_student_text.connect(self.chat_panel.set_live_student)
        self.worker.final_student_text.connect(lambda text: self.chat_panel.append_message("user", text))
        self.worker.live_teacher_text.connect(self.chat_panel.set_live_teacher)
        self.worker.final_teacher_text.connect(lambda text: self.chat_panel.append_message("assistant", text))
        self.worker.emotion_changed.connect(self.status_panel.set_emotion)
        self.worker.session_ready.connect(self.status_panel.set_session)
        self.worker.health_report.connect(self._on_health_report)
        self.worker.error_occurred.connect(self._on_error)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker_thread.start()

    def stop_session(self) -> None:
        if self.worker is not None:
            self.status_panel.set_note("Stopping session...")
            self.worker.stop()
        self.stop_btn.setEnabled(False)

    def _on_finished(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.worker_thread is not None:
            self.worker_thread.wait(2000)
        self.worker_thread = None
        self.worker = None

    def _on_health_report(self, kind: str, value: str) -> None:
        if kind == "TTS":
            self.status_panel.set_tts(value)
        elif kind == "Speaker verification":
            self.status_panel.set_sv(value)
        else:
            self.status_panel.set_note(f"{kind}: {value}")

    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "Session Error", message)
        self.status_panel.set_note(message)
        self.stop_session()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_session()
        super().closeEvent(event)