"""A small cross-process lock on a sidecar file (``<path>.lock``), standard library only.

Used by the spend ledger so that several processes (``bench.run --workers N``) can append
to it and read it at the same time: every append and every read holds the lock, so no
process ever sees a half-written line and no two lines interleave. On Windows the lock is
``msvcrt.locking`` on the first byte of the lock file; elsewhere ``fcntl.flock``. Both are
released by the operating system if the holding process dies.
"""

import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

POLL_SECONDS = 0.01
TIMEOUT_SECONDS = 60.0


class LockTimeoutError(TimeoutError):
    pass


@contextmanager
def locked(path: Path, *, timeout: float = TIMEOUT_SECONDS) -> Iterator[None]:
    """Hold an exclusive lock associated with ``path`` (the file itself is not touched).

    Raises:
        LockTimeoutError: if another process holds the lock for longer than ``timeout``.
    """
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        _acquire(fd, lock_path, timeout)
        try:
            yield
        finally:
            _release(fd)
    finally:
        os.close(fd)


if sys.platform == "win32":
    import msvcrt

    def _acquire(fd: int, lock_path: Path, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() > deadline:
                    raise LockTimeoutError(f"could not lock {lock_path} in {timeout:g} s") from None
                time.sleep(POLL_SECONDS)

    def _release(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _acquire(fd: int, lock_path: Path, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except OSError:
                if time.monotonic() > deadline:
                    raise LockTimeoutError(f"could not lock {lock_path} in {timeout:g} s") from None
                time.sleep(POLL_SECONDS)

    def _release(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
