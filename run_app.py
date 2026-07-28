from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path


def _application_dir() -> Path:
    return Path(__file__).resolve().parent


def _configure_logging() -> Path:
    log_dir = _application_dir() / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        log_dir = Path.home() / "DefectDetectionPlatform" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "startup.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8",
    )
    return log_path


def main():
    log_path = _configure_logging()
    logging.info("Application startup requested from source.")
    try:
        from app.build_info import APP_VERSION
        logging.info("Application version=%s", APP_VERSION)
    except Exception:
        logging.exception("Unable to read application version")

    bundle_root = Path(__file__).resolve().parent
    if str(bundle_root) not in sys.path:
        sys.path.insert(0, str(bundle_root))
    backend_root = bundle_root / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    try:
        from app.main import run
        logging.info("UI module imported; entering Tk main loop.")
        run()
    except Exception:
        logging.exception("Fatal startup error")
        crash_path = log_path.with_name("crash.log")
        crash_path.write_text(traceback.format_exc(), encoding="utf-8")
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Defect Detection Platform",
                "The application could not start.\n\n"
                f"Error details were saved to:\n{crash_path}",
            )
            root.destroy()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
