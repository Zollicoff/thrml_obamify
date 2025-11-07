"""Energy function definition using thrml's EBM framework."""

from __future__ import annotations

from dataclasses import dataclass
import jax.numpy as jnp
from thrml import CategoricalNode, Block
from thrml.models.discrete_ebm import CategoricalEBMFactor


@dataclass
class EnergyParams:
    """Parameters controlling the energy function terms."""
    alpha: float = 0.5  # Weight for content vs style (0=all style, 1=all content)
    lambda_smooth: float = 0.8  # Weight for smoothness/Potts term
    beta: float = 1.0  # Inverse temperature (will be annealed)


def compute_unary_biases(
    source_rgb: jnp.ndarray,
    palette: jnp.ndarray,
    target_hist: jnp.ndarray,
    params: EnergyParams
) -> jnp.ndarray:
    """Compute unary biases for each pixel and each palette color.

    The energy for pixel (i,j) choosing color k is:
        E_unary(i,j,k) = alpha * ||source[i,j] - palette[k]||^2
                         + (1-alpha) * (-log target_hist[k])

    Lower energy = higher probability.

    Args:
        source_rgb: Source image (H, W, 3) in [0,1]
        palette: Color palette (K, 3) in [0,1]
        target_hist: Histogram of palette colors in target (K,)
        params: Energy parameters

    Returns:
        Unary biases array (H, W, K) representing negative energy per state
    """
    height, width = source_rgb.shape[:2]
    num_colors = palette.shape[0]

    # Content term: distance from source pixel to each palette color
    # source_rgb: (H, W, 3)
    # palette: (K, 3)
    # diffs: (H, W, K, 3)
    diffs = source_rgb[:, :, None, :] - palette[None, None, :, :]
    content_energy = jnp.sum(diffs ** 2, axis=-1)  # (H, W, K)

    # Style term: prefer colors frequent in target image
    # -log p(color) = higher energy for rare colors
    eps = 1e-6
    style_energy = -jnp.log(target_hist + eps)  # (K,)
    style_energy = jnp.broadcast_to(style_energy[None, None, :], (height, width, num_colors))

    # Combined energy
    total_energy = params.alpha * content_energy + (1.0 - params.alpha) * style_energy

    # Biases are negative energy (higher bias = lower energy = higher probability)
    biases = -total_energy

    return biases


def create_unary_factor(
    nodes: list[list[CategoricalNode]],
    biases: jnp.ndarray,
    beta: float
) -> CategoricalEBMFactor:
    """Create unary factor for content + style energy.

    Args:
        nodes: 2D grid of CategoricalNode objects
        biases: Unary biases (H, W, K)
        beta: Inverse temperature

    Returns:
        CategoricalEBMFactor for unary terms
    """
    height = len(nodes)
    width = len(nodes[0])

    # Flatten nodes to 1D list
    flat_nodes = [nodes[y][x] for y in range(height) for x in range(width)]

    # Flatten biases to (H*W, K)
    flat_biases = biases.reshape(-1, biases.shape[-1])

    # Scale by temperature
    scaled_biases = beta * flat_biases

    # Create factor with one node group (unary) - wrap in Block
    factor = CategoricalEBMFactor(
        node_groups=[Block(flat_nodes)],
        weights=scaled_biases
    )

    return factor


def create_pairwise_factor(
    neighbor_pairs: list[tuple[CategoricalNode, CategoricalNode]],
    num_colors: int,
    lambda_smooth: float,
    beta: float
) -> CategoricalEBMFactor:
    """Create pairwise Potts factor for smoothness.

    Potts energy: E(i,j) = 0 if colors match, lambda_smooth if different.
    This encourages neighboring pixels to have the same color.

    Args:
        neighbor_pairs: List of (node1, node2) tuples
        num_colors: Number of palette colors (K)
        lambda_smooth: Potts coupling strength
        beta: Inverse temperature

    Returns:
        CategoricalEBMFactor for pairwise smoothness
    """
    # Potts weight matrix: -lambda_smooth on diagonal, 0 off-diagonal
    # (We use negative because factor weights represent -beta * energy)
    potts_weight = jnp.eye(num_colors) * lambda_smooth * beta

    # Create pairwise weights for all edges
    # Shape: (num_edges, K, K)
    num_edges = len(neighbor_pairs)
    pairwise_weights = jnp.tile(potts_weight[None, :, :], (num_edges, 1, 1))

    # Extract node pairs
    nodes1 = [pair[0] for pair in neighbor_pairs]
    nodes2 = [pair[1] for pair in neighbor_pairs]

    # Create pairwise factor - wrap node lists in Blocks
    factor = CategoricalEBMFactor(
        node_groups=[Block(nodes1), Block(nodes2)],
        weights=pairwise_weights
    )

    return factor
