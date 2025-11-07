"""Build thrml graph structure for image pixel grid."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from thrml import CategoricalNode, Block, BlockGibbsSpec


def create_pixel_nodes(height: int, width: int, num_palette_colors: int) -> list[list[CategoricalNode]]:
    """Create a 2D grid of CategoricalNode objects, one per pixel.

    Each pixel can take one of `num_palette_colors` discrete states.

    Args:
        height: Image height in pixels
        width: Image width in pixels
        num_palette_colors: Number of colors in the palette (K)

    Returns:
        2D list of CategoricalNode objects indexed as nodes[y][x]
    """
    nodes = []
    for y in range(height):
        row = []
        for x in range(width):
            node = CategoricalNode()
            row.append(node)
        nodes.append(row)
    return nodes


def create_checkerboard_blocks(nodes: list[list[CategoricalNode]]) -> tuple[list[Block], list[Block]]:
    """Create checkerboard pattern blocks for parallel Gibbs sampling.

    This allows us to update all "even" pixels simultaneously, then all "odd" pixels,
    ensuring no two neighboring pixels are updated at the same time.

    Args:
        nodes: 2D list of CategoricalNode objects

    Returns:
        Tuple of (block_even, block_odd) where each is a list containing one Block
    """
    height = len(nodes)
    width = len(nodes[0])

    even_nodes = []
    odd_nodes = []

    for y in range(height):
        for x in range(width):
            if (y + x) % 2 == 0:
                even_nodes.append(nodes[y][x])
            else:
                odd_nodes.append(nodes[y][x])

    block_even = [Block(even_nodes)]
    block_odd = [Block(odd_nodes)]

    return block_even, block_odd


def get_neighbor_pairs(nodes: list[list[CategoricalNode]]) -> list[tuple[CategoricalNode, CategoricalNode]]:
    """Get all neighboring pixel pairs for defining pairwise energy terms.

    Uses 4-connectivity (up, down, left, right).

    Args:
        nodes: 2D list of CategoricalNode objects

    Returns:
        List of (node1, node2) tuples representing adjacent pixels
    """
    height = len(nodes)
    width = len(nodes[0])
    pairs = []

    for y in range(height):
        for x in range(width):
            current = nodes[y][x]

            # Right neighbor
            if x + 1 < width:
                pairs.append((current, nodes[y][x + 1]))

            # Down neighbor
            if y + 1 < height:
                pairs.append((current, nodes[y + 1][x]))

    return pairs


def create_gibbs_spec(
    free_blocks: list[Block],
    clamped_blocks: list[Block]
) -> BlockGibbsSpec:
    """Create BlockGibbsSpec for thrml sampling.

    Args:
        free_blocks: List of blocks to be sampled
        clamped_blocks: List of blocks to keep fixed (usually empty)

    Returns:
        BlockGibbsSpec for use in sampling
    """
    # Use positional arguments - THRML API requires this
    spec = BlockGibbsSpec(free_blocks, clamped_blocks)

    return spec
