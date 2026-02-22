"""Offline script to build the phonetic IPA index from stations.parquet.

Usage:
    python scripts/build_phonetic_db.py [--stations PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stt.phonetic_db import build_phonetic_index
from src.utils.config import DEFAULT_DATASET_PATH, DEFAULT_PHONETIC_INDEX_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build phonetic IPA index from stations.parquet."
    )
    parser.add_argument(
        "--stations",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to stations.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PHONETIC_INDEX_PATH,
        help="Path to write phonetic_index.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.stations.exists():
        sys.stderr.write(f"Error: stations file not found: {args.stations}\n")
        return 1

    sys.stderr.write(f"Building phonetic index from {args.stations}...\n")
    start = time.perf_counter()
    index = build_phonetic_index(stations_path=args.stations, output_path=args.output)
    elapsed = time.perf_counter() - start

    sys.stderr.write(f"Done: {len(index)} entries in {elapsed:.1f}s\n")
    sys.stderr.write(f"Saved to {args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
