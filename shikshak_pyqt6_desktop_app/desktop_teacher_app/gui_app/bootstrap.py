from __future__ import annotations

import importlib
import sys
from pathlib import Path


def setup_project_imports() -> Path:
    """
    Add sibling project roots so the desktop UI can reuse existing code.

    Expected structure:
        Shikshak/
        ├── shikshak_pyqt6_desktop_app/
        │   └── desktop_teacher_app/
        ├── STT/
        ├── techer_llm/
        └── tts_service/

    All three backends share a top-level ``app`` package with an
    ``app.services`` sub-package.  We merge them into a single
    namespace so all services are importable.

    STT is listed FIRST so that ``app.config`` resolves to the STT
    Settings (which the desktop app depends on for whisper_*, vad_*,
    and teacher_chat_url settings).
    """
    current = Path(__file__).resolve()

    desktop_root = current.parents[1]              # .../desktop_teacher_app
    desktop_wrapper_root = desktop_root.parent      # .../shikshak_pyqt6_desktop_app
    workspace_root = desktop_wrapper_root.parent    # .../Shikshak

    # STT MUST be first — the desktop app's settings come from STT/app/config.py
    sibling_projects = [
        workspace_root / "STT",
        workspace_root / "techer_llm",
        workspace_root / "tts_service",
    ]

    # Insert in REVERSE so the first project (STT) ends up at index 0
    candidates = [desktop_root, desktop_wrapper_root, workspace_root] + sibling_projects
    for path in reversed(candidates):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    # ── Merge ``app`` packages into one namespace ────────────────
    app_dirs = [
        str(project / "app")
        for project in sibling_projects
        if (project / "app").is_dir()
    ]

    if app_dirs:
        # Force-remove any cached app module so it reimports from
        # the corrected sys.path (STT first).
        for key in list(sys.modules):
            if key == "app" or key.startswith("app."):
                del sys.modules[key]

        app_mod = importlib.import_module("app")

        for d in app_dirs:
            if d not in app_mod.__path__:
                app_mod.__path__.append(d)

    # ── Also merge ``app.services`` across all three ─────────────
    svc_dirs = [
        str(project / "app" / "services")
        for project in sibling_projects
        if (project / "app" / "services").is_dir()
    ]

    if svc_dirs:
        if "app.services" in sys.modules:
            svc_mod = sys.modules["app.services"]
        else:
            svc_mod = importlib.import_module("app.services")

        for d in svc_dirs:
            if d not in svc_mod.__path__:
                svc_mod.__path__.append(d)

    return workspace_root