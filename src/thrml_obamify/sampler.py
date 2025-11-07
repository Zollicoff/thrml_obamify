"""Annealed Gibbs sampling using thrml's sample_states() API."""

from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from thrml import CategoricalNode, Block, SamplingSchedule, sample_states
from thrml.factor import FactorSamplingProgram
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
    spec = create_gibbs_spec(free_blocks, clamped_blocks, num_colors)

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

    # Create sampling program
    program = FactorSamplingProgram(
        gibbs_spec=spec,
        factors=[unary_factor, pairwise_factor]
    )

    return program


def initialize_state(
    key: jax.Array,
    nodes: list[list[CategoricalNode]],
    initial_labels: jnp.ndarray,
    free_blocks: list[Block]
) -> dict:
    """Initialize the pixel state for sampling.

    Args:
        key: JAX random key
        nodes: 2D grid of nodes
        initial_labels: Initial pixel labels (H, W)
        free_blocks: Free blocks for initialization

    Returns:
        Initial state dictionary for thrml sampling
    """
    height = len(nodes)
    width = len(nodes[0])

    # Flatten initial labels
    flat_labels = initial_labels.reshape(-1)

    # Create state dict mapping nodes to their initial values
    state = {}
    idx = 0
    for y in range(height):
        for x in range(width):
            state[nodes[y][x]] = flat_labels[idx]
            idx += 1

    return state


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

    # Initialize state
    state = initialize_state(key, nodes, initial_labels, free_blocks)

    frames = []
    current_labels = initial_labels

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

        # Run one sampling step (one pass through all blocks)
        key, subkey = jax.random.split(key)
        thrml_schedule = SamplingSchedule(
            n_warmup=0,
            n_samples=1,
            steps_per_sample=1
        )

        # Sample
        sampled_states = sample_states(
            key=subkey,
            program=program,
            schedule=thrml_schedule,
            init_state=state,
            clamped_states=[],
            observe_blocks=[Block([node for row in nodes for node in row])]
        )

        # Update state for next iteration
        state = sampled_states[-1]  # Get final state

        # Convert state back to labels array
        flat_labels = jnp.array([state[nodes[y][x]] for y in range(height) for x in range(width)])
        current_labels = flat_labels.reshape(height, width)

        # Record frame if needed
        if schedule.record_every and (step + 1) % schedule.record_every == 0:
            frames.append(current_labels)

    return current_labels, frames
