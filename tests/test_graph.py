"""Tests for thrml graph construction."""

import pytest
from thrml import CategoricalNode, Block
from thrml_obamify.graph import (
    create_pixel_nodes,
    create_checkerboard_blocks,
    get_neighbor_pairs,
    create_gibbs_spec
)


def test_create_pixel_nodes():
    """Test that pixel nodes are created correctly."""
    nodes = create_pixel_nodes(height=3, width=4, num_palette_colors=8)

    # Check dimensions
    assert len(nodes) == 3  # height
    assert len(nodes[0]) == 4  # width

    # Check all nodes are CategoricalNode
    for row in nodes:
        for node in row:
            assert isinstance(node, CategoricalNode)


def test_checkerboard_blocks():
    """Test checkerboard block creation."""
    nodes = create_pixel_nodes(height=4, width=4, num_palette_colors=8)
    block_even, block_odd = create_checkerboard_blocks(nodes)

    # Should have one block for even, one for odd
    assert len(block_even) == 1
    assert len(block_odd) == 1

    # Check they are Block objects
    assert isinstance(block_even[0], Block)
    assert isinstance(block_odd[0], Block)

    # Total pixels should equal height * width
    total_pixels = len(block_even[0].nodes) + len(block_odd[0].nodes)
    assert total_pixels == 16


def test_neighbor_pairs():
    """Test neighbor pair extraction."""
    nodes = create_pixel_nodes(height=3, width=3, num_palette_colors=8)
    pairs = get_neighbor_pairs(nodes)

    # For a 3x3 grid with 4-connectivity:
    # Horizontal edges: 3 rows * 2 per row = 6
    # Vertical edges: 2 rows * 3 per row = 6
    # Total = 12
    assert len(pairs) == 12

    # Check pairs are tuples of CategoricalNode
    for n1, n2 in pairs:
        assert isinstance(n1, CategoricalNode)
        assert isinstance(n2, CategoricalNode)
