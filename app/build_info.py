from __future__ import annotations

from pathlib import Path


def _read_version() -> str:
    version_file = Path(__file__).resolve().parents[1] / "VERSION"
    try:
        return version_file.read_text(encoding="utf-8").strip() or "1.0.0"
    except OSError:
        return "1.0.0"


APP_VERSION = _read_version()
DISPLAY_VERSION = APP_VERSION
