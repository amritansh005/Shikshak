from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Bootstrap MUST run before any module that touches app.config
from gui_app.bootstrap import setup_project_imports
setup_project_imports()

from gui_app.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Shikshak AI Teacher")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())