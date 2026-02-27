"""Shared utilities for the book conversion pipeline."""

import base64
import random
from io import BytesIO


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
    n_body  = n - n_front - n_back
    return sorted(
        random.sample(front, min(n_front, len(front))) +
        random.sample(body,  min(n_body,  len(body)))  +
        random.sample(back,  min(n_back,  len(back)))
    )
