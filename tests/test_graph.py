from pathlib import Path

import networkx as nx
import polars as pl
import pytest

from src.pathfinding.algorithm import compute_route
from src.pathfinding.graph import build_graph


def test_build_graph_creates_nodes_and_edges(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "station_id": ["A", "B", "C", "D"],
            "name": ["Alpha", "Bravo", "Charlie", "Delta"],
            "city": ["X", "Y", "Z", "W"],
            "department": ["D1", "D2", "D3", "D4"],
            "lat": [48.0, 48.1, 48.2, 48.3],
            "lon": [2.0, 2.1, 2.2, 2.3],
            "name_norm": ["alpha", "bravo", "charlie", "delta"],
        }
    )
    stations_path = tmp_path / "stations.parquet"
    frame.write_parquet(stations_path)

    graph = build_graph(stations_path=stations_path, k_neighbors=2)

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() > 0
    # Edges should be weighted with positive distances
    weights = [data["weight"] for _, _, data in graph.edges(data=True)]
    assert all(weight > 0 for weight in weights)


def test_compute_route_with_via() -> None:
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=1)
    graph.add_edge("B", "C", weight=1)
    path = compute_route(graph, "A", "C", via=["B"])
    assert path == ["A", "B", "C"]


def test_compute_route_raises_on_missing_path() -> None:
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=1)
    graph.add_node("C")
    with pytest.raises(ValueError):
        compute_route(graph, "A", "C")


def test_build_graph_with_connections(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "station_id": ["A", "B", "C"],
            "name": ["Alpha", "Bravo", "Charlie"],
            "city": ["X", "Y", "Z"],
            "department": ["D1", "D2", "D3"],
            "lat": [48.0, 48.1, 48.2],
            "lon": [2.0, 2.1, 2.2],
            "name_norm": ["alpha", "bravo", "charlie"],
        }
    )
    stations_path = tmp_path / "stations.parquet"
    frame.write_parquet(stations_path)

    connections_path = tmp_path / "connections.csv"
    pl.DataFrame({"from_id": ["A"], "to_id": ["C"], "weight": [10.0]}).write_csv(connections_path)

    graph = build_graph(stations_path=stations_path, connections_path=connections_path, k_neighbors=1)
    assert graph.has_edge("A", "C")
    # ensure kNN not needed for connectivity here
    assert len(graph.edges()) == 1


def test_build_graph_with_schedule(tmp_path: Path) -> None:
    stations = pl.DataFrame(
        {
            "station_id": ["S1", "S2"],
            "name": ["One", "Two"],
            "city": ["X", "Y"],
            "department": ["D1", "D2"],
            "lat": [0.0, 0.1],
            "lon": [0.0, 0.1],
            "name_norm": ["one", "two"],
        }
    )
    stations_path = tmp_path / "stations.parquet"
    stations.write_parquet(stations_path)

    schedule = pl.DataFrame(
        {
            "from_id": ["S1"],
            "to_id": ["S2"],
            "departures_ts": [[1, 2, 3]],
            "durations_sec": [[60, 60, 60]],
        }
    )
    schedule_path = tmp_path / "full_schedule.parquet"
    schedule.write_parquet(schedule_path)

    graph = build_graph(stations_path=stations_path, schedule_path=schedule_path, k_neighbors=1)
    assert graph.has_edge("S1", "S2")
    data = graph.get_edge_data("S1", "S2")
    assert data["departures_ts"] == [1, 2, 3]
