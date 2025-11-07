"""Annealed Gibbs sampling using thrml's sample_states() API."""

from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from thrml import CategoricalNode, Block, SamplingSchedule, sample_states
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import CategoricalGibbsConditional
from .graph import create_gibbs_spec
from .energy import create_unary_factor, create_pairwise_factor, EnergyParams


@dataclass
class AnnealingSchedule:
    """Annealing schedule for temperature control."""
    steps: int  # Total sampling iterations
    beta_start: float  # Initial inverse temperature
    beta_end: float  # Final inverse temperature
    record_every: int = 25  # Save state every N steps


def create_sampling_program(
    nodes: list[list[CategoricalNode]],
    neighbor_pairs: list[tuple[CategoricalNode, CategoricalNode]],
    source_rgb: jnp.ndarray,
    palette: jnp.ndarray,
    target_hist: jnp.ndarray,
    params: EnergyParams,
    free_blocks: list[Block],
    clamped_blocks: list[Block]
) -> FactorSamplingProgram:
    """Create the thrml FactorSamplingProgram with all energy terms.

    Args:
        nodes: 2D grid of pixel nodes
        neighbor_pairs: List of adjacent node pairs
        source_rgb: Source image (H, W, 3)
        palette: Color palette (K, 3)
        target_hist: Target histogram (K,)
        params: Energy parameters
        free_blocks: Blocks to sample
        clamped_blocks: Blocks to keep fixed

    Returns:
        FactorSamplingProgram ready for sampling
    """
    num_colors = palette.shape[0]

    # Create Gibbs spec
    spec = create_gibbs_spec(free_blocks, clamped_blocks)

    # Create unary factor (content + style)
    from .energy import compute_unary_biases
    biases = compute_unary_biases(source_rgb, palette, target_hist, params)
    unary_factor = create_unary_factor(nodes, biases, params.beta)

    # Create pairwise factor (smoothness/Potts)
    pairwise_factor = create_pairwise_factor(
        neighbor_pairs,
        num_colors,
        params.lambda_smooth,
        params.beta
    )

    # Create categorical samplers (one per free block)
    samplers = [CategoricalGibbsConditional(num_colors) for _ in free_blocks]

    # Create sampling program with samplers
    program = FactorSamplingProgram(
        spec,
        samplers,
        [unary_factor, pairwise_factor],
        []  # other_interaction_groups
    )

    return program


def initialize_state(
    key: jax.Array,
    nodes: list[list[CategoricalNode]],
    initial_labels: jnp.ndarray,
    free_blocks: list[Block]
) -> list[jnp.ndarray]:
    """Initialize the pixel state for sampling.

    Args:
        key: JAX random key
        nodes: 2D grid of nodes
        initial_labels: Initial pixel labels (H, W)
        free_blocks: Free blocks for initialization

    Returns:
        List of state arrays (one per free block) for thrml sampling
    """
    height = len(nodes)
    width = len(nodes[0])

    # Flatten initial labels and convert to uint8
    flat_labels = initial_labels.reshape(-1).astype(jnp.uint8)

    # Create node-to-index mapping
    node_to_idx = {}
    idx = 0
    for y in range(height):
        for x in range(width):
            node_to_idx[nodes[y][x]] = idx
            idx += 1

    # Create state array for each free block
    init_states = []
    for block in free_blocks:
        block_indices = [node_to_idx[node] for node in block.nodes]
        block_state = flat_labels[jnp.array(block_indices)]
        init_states.append(block_state)

    return init_states


def run_annealed_sampling(
    key: jax.Array,
    nodes: list[list[CategoricalNode]],
    neighbor_pairs: list[tuple[CategoricalNode, CategoricalNode]],
    source_rgb: jnp.ndarray,
    palette: jnp.ndarray,
    target_hist: jnp.ndarray,
    initial_labels: jnp.ndarray,
    free_blocks: list[Block],
    params: EnergyParams,
    schedule: AnnealingSchedule
) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    """Run annealed Gibbs sampling with thrml.

    Performs simulated annealing by gradually increasing beta (inverse temperature)
    from beta_start to beta_end over the sampling steps.

    Args:
        key: JAX random key
        nodes: 2D grid of pixel nodes
        neighbor_pairs: Adjacent node pairs
        source_rgb: Source image
        palette: Color palette
        target_hist: Target histogram
        initial_labels: Initial state (H, W)
        free_blocks: Blocks to sample
        params: Energy parameters (beta will be overridden during annealing)
        schedule: Annealing schedule

    Returns:
        Tuple of (final_labels, intermediate_frames)
    """
    height = len(nodes)
    width = len(nodes[0])

    # Create beta schedule (linear annealing)
    betas = jnp.linspace(schedule.beta_start, schedule.beta_end, schedule.steps)

    # Initialize state (list of arrays, one per free block)
    init_state_free = initialize_state(key, nodes, initial_labels, free_blocks)

    # Create node-to-index mapping for reconstructing images
    node_to_idx = {}
    idx = 0
    for y in range(height):
        for x in range(width):
            node_to_idx[nodes[y][x]] = idx
            idx += 1

    frames = []
    current_state = init_state_free

    # Annealing loop
    for step in range(schedule.steps):
        # Update beta for this step
        current_params = EnergyParams(
            alpha=params.alpha,
            lambda_smooth=params.lambda_smooth,
            beta=betas[step]
        )

        # Create sampling program with current temperature
        program = create_sampling_program(
            nodes=nodes,
            neighbor_pairs=neighbor_pairs,
            source_rgb=source_rgb,
            palette=palette,
            target_hist=target_hist,
            params=current_params,
            free_blocks=free_blocks,
            clamped_blocks=[]
        )

        # Run one sampling step
        key, subkey = jax.random.split(key)
        thrml_schedule = SamplingSchedule(
            n_warmup=0,
            n_samples=1,
            steps_per_sample=1
        )

        # Sample using correct API: sample_states(key, program, schedule, init_state_free, state_clamp, nodes_to_sample)
        sampled_states = sample_states(
            subkey,
            program,
            thrml_schedule,
            current_state,
            [],  # No clamped states
            free_blocks  # Collect samples from free blocks
        )

        # Update state for next iteration
        # sampled_states is list[Array(n_samples, nodes)] - extract last sample from each block
        current_state = [block_samples[-1] for block_samples in sampled_states]

        # Convert state back to labels array
        flat_labels = jnp.zeros(height * width, dtype=jnp.uint8)
        for block_idx, block in enumerate(free_blocks):
            block_state = current_state[block_idx]
            for node_idx, node in enumerate(block.nodes):
                flat_labels = flat_labels.at[node_to_idx[node]].set(block_state[node_idx])

        current_labels = flat_labels.reshape(height, width)

        # Record frame if needed
        if schedule.record_every and (step + 1) % schedule.record_every == 0:
            frames.append(current_labels)

    return current_labels, frames
