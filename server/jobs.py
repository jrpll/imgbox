"""ML work as jobs: one worker thread runs them in order, so the single-slot model
registry is never raced, and a job outlives the request that queued it — its result
waits at /jobs/{id}/result, for the UI to pick up again after a reload.
"""
import asyncio
import queue
import threading
import traceback
import uuid
from collections import OrderedDict

from fastapi import HTTPException

from device import empty_cache
from progress import Cancelled, tracker

KEPT = 16

_queue: queue.Queue = queue.Queue()
_jobs: OrderedDict = OrderedDict()
_current = None


class Job:
    def __init__(self, fn):
        self.id = uuid.uuid4().hex[:12]
        self.fn = fn
        self.cancelled = False
        self.loop = asyncio.get_running_loop()
        self.future = self.loop.create_future()

    async def wait(self):
        """What fn returned, or the HTTPException it ended with."""
        # Shielded: a waiter that goes away (reload) must not cancel the future for the next one.
        outcome = await asyncio.shield(self.future)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _work():
    global _current
    while True:
        job = _queue.get()
        _current = job
        tracker.cancel.clear()
        try:
            if job.cancelled:
                raise Cancelled()
            tracker.set("Loading", 0, 1)
            outcome = job.fn()
        except Cancelled:
            outcome = HTTPException(status_code=409, detail="cancelled")
        except Exception as e:
            traceback.print_exc()
            outcome = HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
        finally:
            _current = None
            tracker.clear()
            empty_cache()
        job.loop.call_soon_threadsafe(job.future.set_result, outcome)


threading.Thread(target=_work, daemon=True).start()


async def respond(fn, detach: bool = False):
    """Queue fn. Detached, answer at once with the job's id; otherwise with its result."""
    job = Job(fn)
    if detach:
        _jobs[job.id] = job
        while len(_jobs) > KEPT:
            _jobs.popitem(last=False)
    _queue.put(job)
    return {"job_id": job.id} if detach else await job.wait()


def get(id_: str) -> Job:
    if id_ not in _jobs:
        raise HTTPException(status_code=404, detail="unknown job")
    return _jobs[id_]


def cancel(id_: str) -> None:
    """A queued job never starts; the running one stops at its next progress tick."""
    job = get(id_)
    job.cancelled = True
    if job is _current:
        tracker.cancel.set()


def current() -> str | None:
    job = _current
    return job.id if job else None
