from __future__ import annotations

import argparse
from pathlib import Path
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Obamify-style palette sampling with thrml + JAX")
    p.add_argument("--source", type=Path, required=True, help="Path to source image")
    p.add_argument("--target", type=Path, required=True, help="Path to target style image (Obama)")
    p.add_argument("--steps", type=int, default=500, help="Total sampling iterations")
    p.add_argument("--beta_start", type=float, default=0.5, help="Initial inverse temperature")
    p.add_argument("--beta_end", type=float, default=8.0, help="Final inverse temperature")
    p.add_argument("--palette_size", type=int, default=24, help="Number of colors to extract")
    p.add_argument("--alpha", type=float, default=0.5, help="Content vs style weight (0..1)")
    p.add_argument("--lambda_smooth", type=float, default=0.8, help="Potts smoothness weight")
    p.add_argument("--size", type=int, default=None, help="Optional output width (keeps aspect)")
    p.add_argument("--save_gif", action="store_true", help="Save an animated GIF of samples")
    p.add_argument("--output_dir", type=Path, default=Path("output"), help="Output directory")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not (0.0 <= args.alpha <= 1.0):
        raise SystemExit("--alpha must be in [0,1]")
    if args.palette_size < 2:
        raise SystemExit("--palette_size must be >= 2")
    run_pipeline(
        source=args.source,
        target=args.target,
        steps=args.steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        palette_size=args.palette_size,
        alpha=args.alpha,
        lambda_smooth=args.lambda_smooth,
        size=args.size,
        save_gif=args.save_gif,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
