from __future__ import annotations

from os import PathLike
from pathlib import Path


def require_path(value: str | PathLike[str] | None, *, label: str, must_exist: bool = False) -> str:
    """Return a normalized non-empty path or raise a clear configuration error."""
    if value is None or not str(value).strip():
        raise ValueError(f"{label} is required. Select a path in the GUI or pass it explicitly.")

    path = Path(str(value).strip()).expanduser()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)
