"""Tests for CLI interface."""

from thrml_obamify.cli import build_parser


def test_parser_builds():
    """Test that the argument parser builds correctly."""
    parser = build_parser()
    args = parser.parse_args([
        "--source", "assets/image1.png",
        "--target", "assets/obama.jpg",
        "--steps", "10",
    ])
    assert args.steps == 10
    assert str(args.source) == "assets/image1.png"
    assert str(args.target) == "assets/obama.jpg"


def test_parser_with_all_options():
    """Test parser with all options."""
    parser = build_parser()
    args = parser.parse_args([
        "--source", "assets/image1.png",
        "--target", "assets/obama.jpg",
        "--steps", "100",
        "--beta_start", "0.1",
        "--beta_end", "5.0",
        "--palette_size", "16",
        "--alpha", "0.6",
        "--lambda_smooth", "1.0",
        "--size", "128",
        "--save_gif",
        "--seed", "42"
    ])

    assert args.steps == 100
    assert args.beta_start == 0.1
    assert args.beta_end == 5.0
    assert args.palette_size == 16
    assert args.alpha == 0.6
    assert args.lambda_smooth == 1.0
    assert args.size == 128
    assert args.save_gif is True
    assert args.seed == 42
