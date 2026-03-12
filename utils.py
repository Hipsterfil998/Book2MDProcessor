"""Shared utilities for the book conversion pipeline."""

import base64
import contextlib
import os
import random
import sys
import threading
from io import BytesIO
from PIL import Image

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


def pil_to_data_url(img, max_side: int = 1024, jpeg_quality: int = 85) -> str:
    """Encode a PIL image as a base64 JPEG data URL for vLLM multimodal input.

    Resizes to max_side on the longest dimension before encoding to reduce
    token count and payload size without affecting text legibility.
    """
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def sample_indices(total: int, n: int = 20, indices: list[int] | None = None) -> list[int]:
    """Sample page indices for evaluation.

    The first min(10, n) pages are always included (metadata lives there).
    Remaining slots are filled with stratified sampling from body and back.

    If *indices* is provided it is used as the pool instead of range(total),
    allowing blank pages to be excluded from sampling.
    """
    pool = indices if indices is not None else list(range(total))
    total = len(pool)

    if total <= n:
        return list(pool)

    guaranteed = pool[:min(10, n, total)]
    n_remaining = n - len(guaranteed)
    remaining_pool = pool[len(guaranteed):]

    if n_remaining <= 0 or not remaining_pool:
        return sorted(guaranteed)

    back_size = max(1, len(remaining_pool) // 9)   # ~10% of what's left
    back = remaining_pool[-back_size:]
    body = remaining_pool[:-back_size]

    n_back = max(1, n_remaining // 4)
    n_body = n_remaining - n_back

    sampled = (
        random.sample(body, min(n_body, len(body))) +
        random.sample(back, min(n_back, len(back)))
    )
    return sorted(guaranteed + sampled)
