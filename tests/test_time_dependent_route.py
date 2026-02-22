from src.pathfinding.algorithm import compute_earliest_route
import networkx as nx


def test_time_dependent_route_with_connection() -> None:
    g = nx.Graph()
    # A->B departures: 100 (dur 60)
    g.add_edge("A", "B", departures_ts=[100], durations_sec=[60])
    # B->C departures: 200 (dur 60)
    g.add_edge("B", "C", departures_ts=[200], durations_sec=[60])
    # direct A->C dep too late
    g.add_edge("A", "C", departures_ts=[500], durations_sec=[60])

    path, dep_ts, arr_ts = compute_earliest_route(g, "A", "C", departure_ts=90)
    assert path == ["A", "B", "C"]
    assert dep_ts == 100
    assert arr_ts == 260
