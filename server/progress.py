import asyncio
import threading
from dataclasses import dataclass

from tqdm import tqdm


@dataclass(frozen=True)
class Snapshot:
    message: str
    current: int
    total: int


class ProgressTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = Snapshot("", 0, 1)
        self._listeners: list[tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = []

    def set(self, message: str, current: int, total: int) -> None:
        snap = Snapshot(message, current, max(total, 1))
        with self._lock:
            self._snapshot = snap
            listeners = list(self._listeners)
        for queue, loop in listeners:
            loop.call_soon_threadsafe(queue.put_nowait, snap)

    def clear(self) -> None:
        self.set("", 0, 1)

    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snapshot

    def subscribe(self) -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._listeners.append((queue, loop))
            queue.put_nowait(self._snapshot)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._listeners = [(q, l) for q, l in self._listeners if q is not queue]


tracker = ProgressTracker()


def ptqdm(iterable, message: str, total: int | None = None):
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    denom = total if total else 1
    tracker.set(message, 0, denom)
    for i, item in enumerate(tqdm(iterable, total=total, desc=message)):
        yield item
        tracker.set(message, i + 1, denom)
