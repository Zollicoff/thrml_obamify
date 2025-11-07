"""Orchestrate the complete thrml-based Obamify pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as onp
from PIL import Image

from .io import load_image, save_image
from .palette import extract_palette, nearest_palette_indices, palette_histogram
from .graph import create_pixel_nodes, create_checkerboard_blocks, get_neighbor_pairs
from .energy import EnergyParams
from .sampler import AnnealingSchedule, run_annealed_sampling


@dataclass
class PipelineConfig:
    """Configuration for the Obamify pipeline."""
    source: Path
    target: Path
    steps: int = 500
    beta_start: float = 0.5
    beta_end: float = 8.0
    palette_size: int = 24
    alpha: float = 0.5
    lambda_smooth: float = 0.8
    size: int | None = None
    save_gif: bool = False
    output_dir: Path = Path("output")
    seed: int = 0


def apply_palette_to_labels(labels: jnp.ndarray, palette: jnp.ndarray) -> jnp.ndarray:
    """Convert label indices to RGB image.

    Args:
        labels: Label indices (H, W)
        palette: Color palette (K, 3)

    Returns:
        RGB image (H, W, 3)
    """
    return palette[labels]


def run_pipeline(
    source: Path,
    target: Path,
    steps: int,
    beta_start: float,
    beta_end: float,
    palette_size: int,
    alpha: float,
    lambda_smooth: float,
    size: int | None,
    save_gif: bool,
    output_dir: Path,
    seed: int
) -> None:
    """Run the complete thrml-based Obamify pipeline.

    Args:
        source: Path to source image to transform
        target: Path to target Obama image for style
        steps: Number of sampling iterations
        beta_start: Starting inverse temperature
        beta_end: Final inverse temperature
        palette_size: Number of colors to extract from target
        alpha: Content vs style weight (0=style, 1=content)
        lambda_smooth: Smoothness/Potts weight
        size: Optional output width (maintains aspect ratio)
        save_gif: Whether to save animation
        output_dir: Directory for outputs
        seed: Random seed
    """
    print(f"🎨 Starting thrml_obamify pipeline...")
    print(f"   Source: {source}")
    print(f"   Target: {target}")
    print(f"   Steps: {steps}, Beta: {beta_start} → {beta_end}")
    print(f"   Palette: {palette_size} colors, α={alpha}, λ={lambda_smooth}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize RNG
    key = jax.random.PRNGKey(seed)

    # Load images
    print("\n📂 Loading images...")
    src_img = load_image(source, size=size)
    tgt_img = load_image(target, size=size)
    height, width = src_img.shape[:2]
    print(f"   Image size: {width}x{height}")

    # Extract palette from target (Obama)
    print(f"\n🎨 Extracting {palette_size}-color palette from target...")
    palette = extract_palette(tgt_img, k=palette_size)

    # Initialize labels by nearest color in palette
    print("\n🔢 Initializing pixel states...")
    initial_labels = nearest_palette_indices(src_img, palette)

    # Compute target histogram (style prior)
    target_labels = nearest_palette_indices(tgt_img, palette)
    target_hist = palette_histogram(target_labels, palette_size)
    print(f"   Target histogram computed")

    # Build thrml graph structure
    print("\n🕸️  Building thrml graph...")
    nodes = create_pixel_nodes(height, width, palette_size)
    block_even, block_odd = create_checkerboard_blocks(nodes)
    free_blocks = block_even + block_odd  # Alternate between these
    neighbor_pairs = get_neighbor_pairs(nodes)
    print(f"   Created {height}x{width} pixel grid")
    print(f"   {len(neighbor_pairs)} neighbor pairs")
    print(f"   {len(block_even[0].nodes)} even pixels, {len(block_odd[0].nodes)} odd pixels")

    # Set up energy and sampling parameters
    energy_params = EnergyParams(
        alpha=alpha,
        lambda_smooth=lambda_smooth,
        beta=1.0  # Will be overridden by annealing
    )

    annealing_schedule = AnnealingSchedule(
        steps=steps,
        beta_start=beta_start,
        beta_end=beta_end,
        record_every=max(1, steps // 50) if save_gif else 0
    )

    # Run annealed Gibbs sampling with thrml
    print(f"\n🔥 Running annealed Gibbs sampling with thrml...")
    print(f"   Using thrml.sample_states() API")
    start_time = time.time()

    final_labels, frames = run_annealed_sampling(
        key=key,
        nodes=nodes,
        neighbor_pairs=neighbor_pairs,
        source_rgb=src_img,
        palette=palette,
        target_hist=target_hist,
        initial_labels=initial_labels,
        free_blocks=free_blocks,
        params=energy_params,
        schedule=annealing_schedule
    )

    elapsed = time.time() - start_time
    print(f"   ✓ Sampling complete ({elapsed:.1f}s)")

    # Convert labels to RGB
    print("\n🖼️  Rendering final image...")
    final_rgb = apply_palette_to_labels(final_labels, palette)

    # Save final output
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_png = output_dir / f"obamify_{timestamp}.png"
    save_image(final_rgb, output_png)
    print(f"   ✓ Saved: {output_png}")

    # Optionally save GIF
    if save_gif and frames:
        print(f"\n🎬 Creating animation...")
        gif_frames = []
        for lab in frames:
            rgb = apply_palette_to_labels(lab, palette)
            img_arr = onp.clip(onp.asarray(rgb) * 255.0, 0, 255).astype(onp.uint8)
            gif_frames.append(Image.fromarray(img_arr))

        gif_path = output_dir / f"obamify_{timestamp}.gif"
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=80,
            loop=0
        )
        print(f"   ✓ Saved: {gif_path} ({len(gif_frames)} frames)")

    print(f"\n✅ Done! Obama-fied image saved to {output_png}")
