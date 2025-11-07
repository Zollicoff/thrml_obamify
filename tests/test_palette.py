"""Tests for palette extraction."""

import jax.numpy as jnp
import numpy as np
from thrml_obamify.palette import (
    extract_palette,
    nearest_palette_indices,
    palette_histogram
)


def test_extract_palette_shape():
    """Test that palette extraction returns correct shape."""
    # Create a simple test image (4x4 RGB)
    img = jnp.array(np.random.rand(4, 4, 3), dtype=jnp.float32)

    palette = extract_palette(img, k=8)

    # Should return (8, 3) palette
    assert palette.shape == (8, 3)
    # All values should be in [0, 1]
    assert jnp.all(palette >= 0.0)
    assert jnp.all(palette <= 1.0)


def test_nearest_palette_indices():
    """Test nearest palette index assignment."""
    # Create simple 4x4 image
    img = jnp.zeros((4, 4, 3), dtype=jnp.float32)

    # Create palette with 2 colors: black and white
    palette = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=jnp.float32)

    labels = nearest_palette_indices(img, palette)

    # All pixels should map to index 0 (black)
    assert labels.shape == (4, 4)
    assert jnp.all(labels == 0)


def test_palette_histogram():
    """Test histogram computation."""
    # Create labels with known distribution
    labels = jnp.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=jnp.int32)

    hist = palette_histogram(labels, k=2)

    # Should have equal counts for 0 and 1
    assert hist.shape == (2,)
    assert jnp.allclose(hist[0], hist[1], atol=1e-5)
    # Should sum to 1
    assert jnp.allclose(jnp.sum(hist), 1.0)
