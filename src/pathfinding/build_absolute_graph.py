"""Expand GTFS calendar/trips/stop_times into absolute departures per edge.

Output: data/dataset/full_schedule.parquet with columns:
- from_id (str): UIC of origin stop
- to_id (str): UIC of destination stop
- departures_ts (list[int]): Unix timestamps (seconds) of departures
- durations_sec (list[int]): travel durations in seconds aligned with departures_ts
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

RAW_PATH = Path("data/raw")
OUT_PATH = Path("data/dataset")

STOP_TIMES_FILE = RAW_PATH / "stop_times.txt"
TRIPS_FILE = RAW_PATH / "trips.txt"
CALENDAR_DATES_FILE = RAW_PATH / "calendar_dates.txt"  # SNCF privilégie calendar_dates
OUTPUT_FILE = OUT_PATH / "full_schedule.parquet"


def _hhmmss_to_secs(expr: pl.Expr) -> pl.Expr:
    parts = expr.str.split(":")
    h = parts.list.get(0).cast(pl.Int64)
    m = parts.list.get(1).cast(pl.Int64)
    s = parts.list.get(2).cast(pl.Int64).fill_null(0)
    return h * 3600 + m * 60 + s


def build_full_schedule(
    calendar_dates_path: Path = CALENDAR_DATES_FILE,
    trips_path: Path = TRIPS_FILE,
    stop_times_path: Path = STOP_TIMES_FILE,
    output_path: Path = OUTPUT_FILE,
) -> None:
    """Generate absolute departures per edge and persist as Parquet."""
    # 1) Dates actives
    q_dates = (
        pl.scan_csv(calendar_dates_path)
        .filter(pl.col("exception_type") == 1)
        .select(
            [
                pl.col("service_id"),
                pl.col("date").cast(pl.String).str.to_date("%Y%m%d").alias("run_date"),
            ]
        )
    )

    # 2) trips -> service
    q_trips = pl.scan_csv(trips_path).select(["trip_id", "service_id"])

    # 3) Trip x dates
    q_expanded_trips = q_trips.join(q_dates, on="service_id", how="inner").select(["trip_id", "run_date"])

    # 4) stop_times ordonnés + next stop
    q_times = (
        pl.scan_csv(stop_times_path, schema_overrides={"stop_sequence": pl.Int32})
        .select(["trip_id", "stop_id", "departure_time", "arrival_time", "stop_sequence"])
        .with_columns(pl.col("stop_id").str.extract("(\\d+)$", 1).alias("uic"))
        .filter(pl.col("uic").is_not_null())
        .sort(["trip_id", "stop_sequence"])
        .with_columns(
            [
                pl.col("uic").shift(-1).alias("to_id"),
                pl.col("trip_id").shift(-1).alias("next_trip_id"),
                pl.col("arrival_time").shift(-1).alias("next_arr_time"),
            ]
        )
        .filter(
            (pl.col("trip_id") == pl.col("next_trip_id"))
            & (pl.col("uic") != pl.col("to_id"))
            & (pl.col("to_id").is_not_null())
        )
        .select(["trip_id", "uic", "to_id", "departure_time", "next_arr_time"])
    )

    # 5) Jointure trip daté + horaires
    final_data = (
        q_expanded_trips.join(q_times, on="trip_id", how="inner")
        .with_columns(
            [
                (_hhmmss_to_secs(pl.col("departure_time"))).alias("dep_sec"),
                (_hhmmss_to_secs(pl.col("next_arr_time"))).alias("arr_sec"),
            ]
        )
        .with_columns(
            [
                (pl.col("arr_sec") - pl.col("dep_sec")).alias("duration_sec"),
                (
                    (pl.col("run_date").cast(pl.Date).dt.timestamp("ms") // 1000)
                    + pl.col("dep_sec")
                )
                .cast(pl.Int64)
                .alias("ts"),
            ]
        )
        .filter(pl.col("duration_sec") > 0)
        .sort("ts")
    )

    # 6) Agrégation par arc
    graph = (
        final_data.group_by(["uic", "to_id"])
        .agg(
            [
                pl.col("ts").alias("departures_ts"),
                pl.col("duration_sec").alias("durations_sec"),
            ]
        )
        .rename({"uic": "from_id"})
        .collect()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.write_parquet(output_path)
    print(f"✅ Sauvegardé {graph.height} arcs -> {output_path}")


def main() -> None:
    print("📅 Génération du graphe absolu (expansion GTFS)...")
    build_full_schedule()
    print("Terminé.")


if __name__ == "__main__":
    main()
