from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any


class BoundedDropQueue:
    """Single-thread-friendly bounded queue.

    Policy: never drop pending files. When the queue is full, reject the new
    item so the folder watcher can retry it during the next scan.
    """

    def __init__(self, maxsize: int = 32):
        self.maxsize = max(1, int(maxsize))
        self._items = deque()
        self.dropped_count = 0

    def put(self, item: Any) -> bool:
        if len(self._items) >= self.maxsize:
            return False
        self._items.append(item)
        return True

    def get(self) -> Any:
        return self._items.popleft()

    def empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()


class DuplicateFileTracker:
    def __init__(self):
        self._seen: set[tuple[str, int, int]] = set()

    def should_process(self, path: str | Path) -> bool:
        signature = self._signature(path)
        if signature in self._seen:
            return False
        self._seen.add(signature)
        return True

    def forget(self, path: str | Path) -> None:
        """Allow a file to be discovered again when it was not queued."""
        try:
            signature = self._signature(path)
        except OSError:
            return
        self._seen.discard(signature)

    def clear(self) -> None:
        self._seen.clear()

    @staticmethod
    def _signature(path: str | Path) -> tuple[str, int, int]:
        file_path = Path(path).expanduser().resolve()
        stat = file_path.stat()
        return (str(file_path).casefold(), int(stat.st_mtime_ns), int(stat.st_size))


class RecentResultBuffer:
    def __init__(self, maxlen: int = 200):
        self.maxlen = max(1, int(maxlen))
        self._items = deque(maxlen=self.maxlen)

    def append(self, item: Any) -> None:
        self._items.append(item)

    def clear(self) -> None:
        self._items.clear()

    def rows(self) -> list[Any]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


def release_camera_handle(handle: Any) -> None:
    if handle is None:
        return
    release = getattr(handle, "release", None)
    if callable(release):
        release()


def can_start_worker(worker: Any) -> bool:
    return not (worker is not None and callable(getattr(worker, "is_alive", None)) and worker.is_alive())


def format_inference_error(path: str | Path, exc: BaseException) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return f"[Warning] Image not ready or invalid, will retry: {path} ({message})"
