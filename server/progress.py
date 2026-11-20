import asyncio
import threading
from dataclasses import dataclass

from tqdm import tqdm


@dataclass(frozen=True)
class Snapshot:
    message: str
    current: int
    total: int
    remaining: str = ""


class Cancelled(Exception):
    """Raised inside the running job, at its next progress tick, once it has been cancelled."""


class ProgressTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = Snapshot("", 0, 1, "")
        self._listeners: list[tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = []
        self.cancel = threading.Event()

    def set(self, message: str, current: int, total: int, remaining: str = "") -> None:
        if self.cancel.is_set():
            raise Cancelled()
        self._publish(Snapshot(message, current, max(total, 1), remaining))

    def _publish(self, snap: Snapshot) -> None:
        with self._lock:
            self._snapshot = snap
            listeners = list(self._listeners)
        for queue, loop in listeners:
            loop.call_soon_threadsafe(queue.put_nowait, snap)

    def set_from_tqdm(self, t: tqdm) -> None:
        fd = t.format_dict
        n = fd.get("n") or 0
        total = fd.get("total") or 1
        rate = fd.get("rate")
        remaining = ""
        if rate and rate > 0 and total and n < total:
            remaining = tqdm.format_interval((total - n) / rate)
        self.set(t.desc or "", n, total, remaining)

    def clear(self) -> None:
        self._publish(Snapshot("", 0, 1, ""))

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
    t = tqdm(iterable, total=total, desc=message)
    for item in t:
        yield item
        tracker.set_from_tqdm(t)
