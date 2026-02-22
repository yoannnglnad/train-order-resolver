"""Graph construction utilities for SNCF network data."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Iterable, Tuple

import networkx as nx
import polars as pl

from src.utils.config import (
    DEFAULT_CONNECTIONS_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_SCHEDULE_PATH,
    DEFAULT_K_NEIGHBORS,
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two WGS84 points."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _validate_columns(frame: pl.DataFrame) -> None:
    required = {"station_id", "name", "city", "department", "lat", "lon", "name_norm"}
    missing = required.difference(set(frame.columns))
    if missing:
        raise ValueError(f"Missing required station columns: {', '.join(sorted(missing))}")


def _iter_neighbors(
    stations: pl.DataFrame, k_neighbors: int
) -> Iterable[Tuple[str, Iterable[Tuple[str, float]]]]:
    ids = stations["station_id"].to_list()
    lats = stations["lat"].to_list()
    lons = stations["lon"].to_list()
    n = len(ids)
    for i in range(n):
        distances: list[Tuple[str, float]] = []
        for j in range(n):
            if i == j:
                continue
            dist = haversine_km(lats[i], lons[i], lats[j], lons[j])
            distances.append((ids[j], dist))
        distances.sort(key=lambda x: x[1])
        yield ids[i], distances[:k_neighbors]


def _load_connections(
    stations: pl.DataFrame, connections_path: Path
) -> Iterable[Tuple[str, str, float]]:
    """Yield edges (u, v, weight) from connections file; compute haversine if weight absent."""
    path = Path(connections_path)
    if not path.exists():
        return []

    if path.suffix.lower() == ".csv":
        frame = pl.read_csv(path)
    else:
        frame = pl.read_parquet(path)

    required = {"from_id", "to_id"}
    missing = required.difference(set(frame.columns))
    if missing:
        raise ValueError(f"Connections file missing columns: {', '.join(sorted(missing))}")

    stations_map = stations.select(["station_id", "lat", "lon"]).to_dict(as_series=False)
    lat_map = dict(zip(stations_map["station_id"], stations_map["lat"]))
    lon_map = dict(zip(stations_map["station_id"], stations_map["lon"]))

    edges: list[Tuple[str, str, float]] = []
    for row in frame.to_dicts():
        u = row["from_id"]
        v = row["to_id"]
        weight = row.get("weight")
        if weight is None:
            weight = haversine_km(lat_map[u], lon_map[u], lat_map[v], lon_map[v])
        edges.append((u, v, float(weight)))
    return edges


def _load_schedule(schedule_path: Path) -> Iterable[Tuple[str, str, list[int], list[int], float]]:
    """Yield edges with departures/durations; also compute a representative weight."""
    if not schedule_path.exists():
        return []
    frame = pl.read_parquet(schedule_path)
    required = {"from_id", "to_id", "departures_ts", "durations_sec"}
    missing = required.difference(set(frame.columns))
    if missing:
        raise ValueError(f"Schedule file missing columns: {', '.join(sorted(missing))}")

    for row in frame.to_dicts():
        dep_list = row["departures_ts"]
        dur_list = row["durations_sec"]
        if not dep_list or not dur_list:
            continue
        # Representative weight: median duration in minutes
        sorted_dur = sorted(dur_list)
        median_dur = sorted_dur[len(sorted_dur) // 2] / 60.0
        yield row["from_id"], row["to_id"], dep_list, dur_list, median_dur


def _add_city_transfers(
    graph: nx.Graph, frame: pl.DataFrame, transfer_weight_min: float = 30.0,
    max_distance_km: float = 30.0,
) -> None:
    """Add transfer edges between stations in the same city.

    This is critical for cities with multiple terminus stations (e.g. Paris)
    where passengers can transfer by foot or metro between gares.
    Transfer edges are always available with a fixed duration, so they carry
    no schedule data — the routing algorithm handles them specially.

    A geographic distance cap filters out homonymous communes
    (e.g. two different "Dommartin" 500 km apart).
    """
    skip_cities = {"INCONNU", ""}
    city_groups: dict[str, list[str]] = {}
    for row in frame.to_dicts():
        city = row["city"].strip().upper()
        if city in skip_cities:
            continue
        sid = row["station_id"]
        if sid in graph:
            city_groups.setdefault(city, []).append(sid)

    for city, sids in city_groups.items():
        if len(sids) < 2:
            continue
        for i, u in enumerate(sids):
            for v in sids[i + 1 :]:
                if graph.has_edge(u, v):
                    continue
                lat_u = graph.nodes[u].get("lat")
                lon_u = graph.nodes[u].get("lon")
                lat_v = graph.nodes[v].get("lat")
                lon_v = graph.nodes[v].get("lon")
                if lat_u and lon_u and lat_v and lon_v:
                    if haversine_km(lat_u, lon_u, lat_v, lon_v) > max_distance_km:
                        continue
                graph.add_edge(
                    u,
                    v,
                    weight=transfer_weight_min,
                    is_transfer=True,
                )


def build_graph(
    stations_path: Path | None = None,
    connections_path: Path | None = None,
    schedule_path: Path | None = None,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
) -> nx.Graph:
    """
    Load station data and build a weighted undirected graph.

    Nodes carry station attributes; edges connect each station to its k nearest
    neighbors using haversine distance (km) as weight. This approximates network
    connectivity when explicit timetable edges are absent.
    """
    stations_path = stations_path or DEFAULT_DATASET_PATH
    schedule_path = schedule_path or DEFAULT_SCHEDULE_PATH

    # Only use pickle cache for default dataset paths
    use_cache = (
        stations_path == DEFAULT_DATASET_PATH
        and schedule_path == DEFAULT_SCHEDULE_PATH
    )
    cache_path = Path("data/cache/graph.pickle")

    if use_cache and cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        sources_valid = True
        for src in [stations_path, schedule_path]:
            if src and src.exists() and src.stat().st_mtime > cache_mtime:
                sources_valid = False
                break
        if sources_valid:
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    frame = pl.read_parquet(stations_path)
    _validate_columns(frame)
    graph = nx.Graph()

    station_ids = set()
    for row in frame.to_dicts():
        node_id = row["station_id"]
        station_ids.add(node_id)
        graph.add_node(
            node_id,
            name=row["name"],
            city=row["city"],
            department=row["department"],
            lat=row["lat"],
            lon=row["lon"],
            name_norm=row["name_norm"],
        )

    # Prefer absolute schedule if provided, then explicit connections, otherwise kNN
    edges_added = False
    connections_path = connections_path or DEFAULT_CONNECTIONS_PATH

    schedule_edges = list(_load_schedule(schedule_path)) if schedule_path else []
    for u, v, departures, durations, w in schedule_edges:
        if u not in station_ids or v not in station_ids:
            continue
        graph.add_edge(u, v, weight=w, departures_ts=departures, durations_sec=durations)
        edges_added = True

    if not edges_added and connections_path:
        connections = list(_load_connections(frame, connections_path)) if connections_path else []
        for u, v, w in connections:
            if u not in station_ids or v not in station_ids:
                continue
            graph.add_edge(u, v, weight=w)
            edges_added = True

    if not edges_added:
        for source_id, neighbors in _iter_neighbors(frame, k_neighbors=k_neighbors):
            for target_id, distance in neighbors:
                if graph.has_edge(source_id, target_id):
                    continue
                graph.add_edge(source_id, target_id, weight=distance)

    # Always add intra-city transfer edges (e.g. Paris-St-Lazare ↔ Paris-Est)
    _add_city_transfers(graph, frame, transfer_weight_min=30.0)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    return graph
