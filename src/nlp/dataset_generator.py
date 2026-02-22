"""Synthetic dataset generator for travel order sentences."""

from __future__ import annotations

import argparse
from pathlib import Path


def generate_dataset(count: int, output_path: Path) -> None:
    """Generate a placeholder dataset; replace with real templates and sampling logic."""
    raise NotImplementedError("Dataset generation is not implemented yet.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic travel order dataset.")
    parser.add_argument("--count", type=int, default=10_000, help="Number of sentences to generate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dataset/train.csv"),
        help="Target CSV path for the generated dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(args.count, args.output)


if __name__ == "__main__":
    main()
