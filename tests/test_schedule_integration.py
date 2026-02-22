from pathlib import Path

import polars as pl

from src.pathfinding.build_absolute_graph import build_full_schedule, OUTPUT_FILE
from src.pathfinding.graph import build_graph
from src.pathfinding.algorithm import next_departure_on_edge
from src.nlp.inference import TravelResolver


def test_full_schedule_generation_and_lookup(tmp_path: Path, monkeypatch) -> None:
    # Create minimal GTFS files
    raw = tmp_path / "raw"
    raw.mkdir()
    # calendar_dates: service 1 runs on 2026-01-02
    (raw / "calendar_dates.txt").write_text("service_id,date,exception_type\n1,20260102,1\n", encoding="utf-8")
    # trips: one trip for service 1
    (raw / "trips.txt").write_text("route_id,service_id,trip_id,trip_headsign,direction_id,block_id,shape_id\nR,1,T1,0,0,0,\n", encoding="utf-8")
    # stop_times: two stops with 08:00 -> 08:05
    (raw / "stop_times.txt").write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence,stop_headsign,pickup_type,drop_off_type,shape_dist_traveled\n"
        "T1,08:00:00,08:00:00,StopPoint:000001,1,,0,0,\n"
        "T1,08:05:00,08:05:00,StopPoint:000002,2,,0,0,\n",
        encoding="utf-8",
    )

    out_dataset = tmp_path / "dataset"
    out_dataset.mkdir()
    monkeypatch.setattr("src.pathfinding.build_absolute_graph.RAW_PATH", raw)
    monkeypatch.setattr("src.pathfinding.build_absolute_graph.OUT_PATH", out_dataset)
    monkeypatch.setattr("src.pathfinding.build_absolute_graph.STOP_TIMES_FILE", raw / "stop_times.txt")
    monkeypatch.setattr("src.pathfinding.build_absolute_graph.TRIPS_FILE", raw / "trips.txt")
    monkeypatch.setattr("src.pathfinding.build_absolute_graph.CALENDAR_DATES_FILE", raw / "calendar_dates.txt")
    monkeypatch.setattr("src.pathfinding.build_absolute_graph.OUTPUT_FILE", out_dataset / "full_schedule.parquet")

    build_full_schedule(
        calendar_dates_path=raw / "calendar_dates.txt",
        trips_path=raw / "trips.txt",
        stop_times_path=raw / "stop_times.txt",
        output_path=out_dataset / "full_schedule.parquet",
    )

    df = pl.read_parquet(out_dataset / "full_schedule.parquet")
    assert df.height == 1
    row = df.to_dicts()[0]
    assert row["from_id"] == "000001" and row["to_id"] == "000002"
    assert row["departures_ts"] and row["durations_sec"] == [300]

    # Build graph using the generated schedule
    stations = pl.DataFrame(
        {
            "station_id": ["000001", "000002"],
            "name": ["A", "B"],
            "city": ["X", "Y"],
            "department": ["D1", "D2"],
            "lat": [0.0, 0.1],
            "lon": [0.0, 0.1],
            "name_norm": ["a", "b"],
        }
    )
    stations_path = out_dataset / "stations.parquet"
    stations.write_parquet(stations_path)

    schedule_path = out_dataset / "full_schedule.parquet"
    graph = build_graph(stations_path=stations_path, schedule_path=schedule_path, k_neighbors=1)
    dep_ts, dur = next_departure_on_edge(graph, "000001", "000002", row["departures_ts"][0] - 10)
    assert dep_ts == row["departures_ts"][0]
    assert dur == 300

    resolver = TravelResolver(
        stations_path=stations_path,
        connections_path=None,
        schedule_path=schedule_path,
        k_neighbors=1,
        cache=None,
        logger=None,
    )
    order = resolver.resolve_order("1", "Je veux aller de a à b", target_ts=row["departures_ts"][0] - 10)
    assert order.is_valid
    assert order.departure_ts == row["departures_ts"][0]
    assert abs(order.duration_min - 5.0) < 1e-6
