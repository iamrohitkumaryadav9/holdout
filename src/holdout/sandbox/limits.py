"""Resource limits for sandboxed processes, behind one interface for both platforms.

* Linux: ``RLIMIT_DATA`` (memory) and ``RLIMIT_CPU`` are set in the child before it runs,
  so the kernel enforces them; the psutil watchdog also runs, as on Windows.
* Windows: there is no ``resource`` module; a psutil watchdog thread samples the child's
  resident memory and kills it when it exceeds the limit. Sampling can miss a very fast
  spike between samples; that is the honest limit of a watchdog.

The wall-clock timeout is enforced by the caller with ``subprocess`` timeouts on both.
"""

import importlib
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import psutil


@dataclass(frozen=True)
class Limits:
    timeout_seconds: float = 20.0
    memory_bytes: int = 1024**3
    output_bytes: int = 64 * 1024


def preexec_for(limits: Limits) -> Callable[[], None] | None:
    """A ``preexec_fn`` applying kernel limits on POSIX; None on Windows."""
    if os.name == "nt":
        return None

    def apply() -> None:  # pragma: no cover - runs in the child process on POSIX
        # Imported dynamically: the module (and its RLIMIT_* names) exists on POSIX only.
        resource: Any = importlib.import_module("resource")
        resource.setrlimit(resource.RLIMIT_DATA, (limits.memory_bytes, limits.memory_bytes))
        cpu = int(limits.timeout_seconds) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))

    return apply


@dataclass
class MemoryWatchdog:
    """Kill ``pid`` (and its children) if resident memory exceeds the limit."""

    pid: int
    memory_bytes: int
    interval_seconds: float = 0.05
    tripped: bool = False
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> "MemoryWatchdog":
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _watch(self) -> None:
        try:
            process = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return
        while not self._stop.is_set():
            try:
                rss = process.memory_info().rss + sum(
                    c.memory_info().rss for c in process.children(recursive=True)
                )
            except psutil.NoSuchProcess:
                return
            if rss > self.memory_bytes:
                self.tripped = True
                for child in process.children(recursive=True):
                    child.kill()
                process.kill()
                return
            self._stop.wait(self.interval_seconds)
