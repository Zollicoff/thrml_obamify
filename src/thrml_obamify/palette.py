from __future__ import annotations

import numpy as onp
from PIL import Image
import jax.numpy as jnp


def extract_palette(img_rgb01: jnp.ndarray, k: int) -> jnp.ndarray:
    # Use Pillow's quantize to get a K-color palette from target
    arr = (onp.asarray(img_rgb01) * 255.0).astype(onp.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    quant = pil.quantize(colors=k, method=Image.MEDIANCUT)
    pal = quant.getpalette()[: 3 * k]
    palette = onp.array(pal, dtype=onp.float32).reshape(-1, 3) / 255.0
    return jnp.array(palette)  # (K,3)


def nearest_palette_indices(img_rgb01: jnp.ndarray, palette: jnp.ndarray) -> jnp.ndarray:
    # Assign each pixel to nearest palette color (L2 in RGB)
    h, w, _ = img_rgb01.shape
    x = img_rgb01.reshape(-1, 3)  # (N,3)
    diffs = x[:, None, :] - palette[None, :, :]
    d2 = jnp.sum(diffs * diffs, axis=-1)  # (N,K)
    labels = jnp.argmin(d2, axis=1).reshape(h, w)
    return labels


def palette_histogram(labels: jnp.ndarray, k: int, eps: float = 1e-6) -> jnp.ndarray:
    counts = jnp.bincount(labels.reshape(-1), length=k)
    probs = (counts + eps) / (jnp.sum(counts) + eps * k)
    return probs
