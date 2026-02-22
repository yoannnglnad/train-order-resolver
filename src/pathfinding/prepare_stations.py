"""Convert SNCF station parquet into a curated Parquet for graph building and NLP matching."""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import List

import polars as pl

if __package__ is None or __package__ == "":
    # Allow running as a script: python src/pathfinding/prepare_stations.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.config import DEFAULT_DATA_DIR, DEFAULT_RAW_DATA_DIR


def normalize_name(value: str) -> str:
    """Lowercase, strip accents, and collapse punctuation for matching."""
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    cleaned = stripped.replace("-", " ").replace("'", " ").replace(".", " ")
    return " ".join(cleaned.lower().split())


def select_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Filter, clean, and reshape the station dataset."""
    working = (
        frame
        .filter(pl.col("voyageurs").cast(str).str.to_uppercase() == "O")
        .drop_nulls(subset=["x_wgs84", "y_wgs84"])
        .with_columns(
            [
                pl.col("code_uic").cast(str).alias("station_id"),
                pl.col("libelle").cast(str).str.strip_chars().alias("name"),
                pl.col("commune").cast(str).str.strip_chars().alias("city"),
                pl.col("departemen").cast(str).str.strip_chars().alias("department"),
                pl.col("y_wgs84").cast(float).alias("lat"),
                pl.col("x_wgs84").cast(float).alias("lon"),
                pl.col("voyageurs").cast(str).str.to_uppercase().alias("passengers"),
                pl.col("fret").cast(str).str.to_uppercase().alias("freight"),
            ]
        )
    )

    working = working.with_columns(
        pl.col("name").map_elements(normalize_name, return_dtype=pl.String).alias("name_norm")
    )

    working = working.unique(subset=["station_id"])

    cols: List[str] = [
        "station_id",
        "name",
        "city",
        "department",
        "lat",
        "lon",
        "passengers",
        "freight",
        "name_norm",
    ]
    return working.select(cols)


def prepare_stations(input_path: Path, output_path: Path) -> None:
    """Load parquet, select useful columns, and persist as Parquet."""
    frame = pl.read_parquet(input_path)
    curated = select_columns(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    curated.write_parquet(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare curated station dataset from SNCF parquet export."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR / "liste-des-gares.parquet",
        help="Path to the source parquet file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_DIR / "dataset" / "stations.parquet",
        help="Path to write the curated Parquet.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepare_stations(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
