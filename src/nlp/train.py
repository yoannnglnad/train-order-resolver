"""Training entrypoint for the CamemBERT-based NER model."""

from __future__ import annotations

import argparse
from pathlib import Path


def train_model(dataset_path: Path, epochs: int) -> None:
    """Train the model on the provided dataset."""
    raise NotImplementedError("Model training is not implemented yet.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NER model for travel order parsing.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/dataset/train.csv"),
        help="Path to the training dataset.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(args.dataset, args.epochs)


if __name__ == "__main__":
    main()
