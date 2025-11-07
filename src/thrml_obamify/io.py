from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as onp
from PIL import Image
import jax.numpy as jnp


def load_image(path: Path, size: Optional[int] = None) -> jnp.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None:
        w, h = img.size
        new_w = size
        new_h = int(h * (new_w / w))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    arr = (onp.asarray(img).astype(onp.float32) / 255.0)
    return jnp.array(arr)


def save_image(array: jnp.ndarray, path: Path) -> None:
    arr = onp.clip(onp.asarray(array) * 255.0, 0, 255).astype(onp.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
