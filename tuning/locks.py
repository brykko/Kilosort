"""Cross-process exclusive lock on a file, for running two search jobs side by side: the GPU-heavy
KS4 stages (stage_cache.py) and the allocation of run numbers (run_candidate.py) take one at a time.
The lock is released when the holder exits, even if it crashes."""
import os
import time
from contextlib import contextmanager
from pathlib import Path

if os.name == 'nt':
    import msvcrt

    def _try(fd):
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _release(fd):
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try(fd):
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _release(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def exclusive(path, poll_s=0.5):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        while True:
            os.lseek(fd, 0, os.SEEK_SET)
            if _try(fd):
                break
            time.sleep(poll_s)
        try:
            yield
        finally:
            _release(fd)
    finally:
        os.close(fd)
