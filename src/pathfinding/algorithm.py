"""Shortest-path algorithms for travel itineraries."""

from __future__ import annotations

import bisect
import heapq
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx


def compute_route(
    graph: nx.Graph,
    departure: str,
    destination: str,
    via: Sequence[str] | None = None,
) -> List[str]:
    """
    Return the ordered list of station_ids from departure to destination.

    - Validates that all nodes exist in the graph.
    - Supports optional `via` sequence; computes concatenated shortest paths.
    - Raises ValueError when nodes are unknown or when no path exists.
    """
    if departure not in graph or destination not in graph:
        raise ValueError("Departure or destination not found in graph.")

    stops: list[str] = []
    current = departure
    via_sequence: Iterable[str] = via or []

    for target in [*via_sequence, destination]:
        if target not in graph:
            raise ValueError(f"Via/destination node not found in graph: {target}")
        try:
            segment = nx.shortest_path(graph, source=current, target=target, weight="weight")
        except nx.NetworkXNoPath as exc:
            raise ValueError(f"No path found between {current} and {target}") from exc
        if stops:
            segment = segment[1:]
        stops.extend(segment)
        current = target

    return stops


def compute_route_with_exploration(
    graph: nx.Graph,
    departure: str,
    destination: str,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Compute shortest path using manual Dijkstra, tracking all explored edges.

    Returns:
        (path, explored_edges) where:
        - path: List[str] of station_ids from departure to destination
        - explored_edges: List[(from_id, to_id)] in exploration order
    """
    if departure not in graph:
        raise ValueError(f"Departure not found in graph: {departure}")
    if destination not in graph:
        raise ValueError(f"Destination not found in graph: {destination}")

    dist: Dict[str, float] = {departure: 0.0}
    prev: Dict[str, str] = {}
    explored_edges: List[Tuple[str, str]] = []
    heap: list[Tuple[float, str]] = [(0.0, departure)]

    while heap:
        d_u, u = heapq.heappop(heap)
        if u == destination:
            break
        if d_u != dist.get(u):
            continue  # stale entry
        for v in graph.neighbors(u):
            explored_edges.append((u, v))
            weight = graph[u][v].get("weight", 1.0)
            d_v = d_u + weight
            if d_v < dist.get(v, float("inf")):
                dist[v] = d_v
                prev[v] = u
                heapq.heappush(heap, (d_v, v))

    if destination not in prev and departure != destination:
        raise ValueError(f"No path found between {departure} and {destination}")

    # Reconstruct path
    path: List[str] = []
    node = destination
    while True:
        path.append(node)
        if node == departure:
            break
        node = prev[node]
    path.reverse()

    return path, explored_edges


def next_departure_on_edge(
    graph: nx.Graph, u: str, v: str, target_ts: int
) -> Tuple[int, int]:
    """
    Return (departure_ts, duration_sec) for the next train on edge (u, v) after target_ts.
    Raises ValueError if no schedule is available or no future departure exists.
    """
    if not graph.has_edge(u, v):
        raise ValueError(f"Edge {u}-{v} missing in graph")
    data = graph.get_edge_data(u, v)
    departures: list[int] | None = data.get("departures_ts")
    durations: list[int] | None = data.get("durations_sec")
    if not departures or not durations:
        raise ValueError("No schedule data on this edge")
    idx = bisect.bisect_left(departures, target_ts)
    if idx >= len(departures):
        raise ValueError("No future departure found for requested time")
    return departures[idx], durations[idx]


def compute_earliest_route(
    graph: nx.Graph,
    source: str,
    target: str,
    departure_ts: int,
    transfer_buffer_sec: int = 0,
    buffer_fn: callable | None = None,
) -> Tuple[List[str], int, int]:
    """
    Time-dependent earliest-arrival routing using schedule on edges.

    Returns (path, first_departure_ts, arrival_ts).
    Raises ValueError if no path with feasible departures exists.
    """
    if source not in graph or target not in graph:
        raise ValueError("Source or target not in graph.")

    # dist holds best arrival time at node
    dist: Dict[str, int] = {source: departure_ts}
    prev: Dict[str, str] = {}
    first_dep: Dict[str, int] = {}
    heap = [(departure_ts, source)]

    while heap:
        arr_u, u = heapq.heappop(heap)
        if u == target:
            break
        if arr_u != dist.get(u, None):
            continue  # stale
        for v in graph.neighbors(u):
            data = graph.get_edge_data(u, v)
            if data.get("is_transfer"):
                # Transfer: always available, fixed duration
                transfer_sec = int(data["weight"] * 60)
                dep_ts = arr_u
                dur_sec = transfer_sec
                arr_v = dep_ts + dur_sec
            else:
                # Normal edge: use schedule
                departures = data.get("departures_ts")
                durations = data.get("durations_sec")
                if not departures or not durations:
                    continue
                buff = buffer_fn(u, v) if buffer_fn else transfer_buffer_sec
                try:
                    dep_ts, dur_sec = next_departure_on_edge(graph, u, v, arr_u + buff)
                except ValueError:
                    continue
                arr_v = dep_ts + dur_sec
            if arr_v < dist.get(v, float("inf")):
                dist[v] = arr_v
                prev[v] = u
                first_dep[v] = first_dep.get(u, dep_ts)
                heapq.heappush(heap, (arr_v, v))

    if target not in dist:
        raise ValueError("No feasible timed path found.")

    # reconstruct path
    path: List[str] = []
    node = target
    while True:
        path.append(node)
        if node == source:
            break
        node = prev[node]
    path.reverse()
    return path, first_dep.get(target, departure_ts), dist[target]
