"""Shared utilities for the book conversion pipeline."""

import base64
import contextlib
import os
import random
import sys
import threading
from io import BytesIO

# Patterns emitted by grpc/protobuf 3→4 incompatibility in vLLM worker processes
_PROTOBUF_NOISE = (b"GetPrototype", b"MessageFactory")


@contextlib.contextmanager
def suppress_worker_stderr():
    """Filter OS-level stderr to drop protobuf MessageFactory noise from vLLM workers.

    Worker processes inherit fd 2 directly, so a Python sys.stderr wrapper is not enough.
    This context manager intercepts fd 2 with a pipe, drops matching lines, and forwards
    everything else to the real stderr.
    """
    real_fd = os.dup(2)
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    os.close(w_fd)

    def _pump():
        buf = b""
        try:
            while True:
                try:
                    chunk = os.read(r_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not any(p in line for p in _PROTOBUF_NOISE):
                        os.write(real_fd, line + b"\n")
            if buf and not any(p in buf for p in _PROTOBUF_NOISE):
                os.write(real_fd, buf)
        except OSError:
            pass
        finally:
            try:
                os.close(r_fd)
            except OSError:
                pass

    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    try:
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(real_fd, 2)   # restoring fd 2 closes the write end → pump thread sees EOF
        os.close(real_fd)
        t.join(timeout=5)


def pil_to_data_url(img) -> str:
    """Encode a PIL image as a base64 PNG data URL for vLLM multimodal input."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def sample_indices(total: int, n: int = 20) -> list[int]:
    """Stratified sampling across front / body / back of the document."""
    if total <= n:
        return list(range(total))
    front = list(range(0, max(1, total // 10)))
    back  = list(range(total - max(1, total // 10), total))
    body  = list(range(len(front), total - len(back)))
    n_front = max(1, n // 7)
    n_back  = max(1, n // 7)
    if n_front + n_back > n:          # guard for very small n
        n_front = n // 2
        n_back  = n - n_front
    n_body = n - n_front - n_back
    return sorted(
        random.sample(front, min(n_front, len(front))) +
        random.sample(body,  min(n_body,  len(body)))  +
        random.sample(back,  min(n_back,  len(back)))
    )
