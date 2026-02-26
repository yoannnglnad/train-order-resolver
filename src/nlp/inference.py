"""Inference helpers for extracting travel intents and entities from text."""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import polars as pl
from difflib import SequenceMatcher

from src.pathfinding.algorithm import compute_route, compute_route_with_exploration
from src.pathfinding.algorithm import compute_earliest_route
from src.pathfinding.graph import build_graph
from src.pathfinding.prepare_stations import normalize_name
from src.utils.cache import Cache
from src.utils.config import (
    DEFAULT_CONNECTIONS_PATH,
    DEFAULT_CONNECTION_RULES_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_K_NEIGHBORS,
)
from src.utils.logging import log_decision
from src.nlp.spacy_extractor import SpacyExtractor
from src.nlp.hf_inference import HFExtractor


@dataclass
class TravelOrder:
    sentence_id: str
    departure_id: Optional[str]
    arrival_id: Optional[str]
    via_ids: List[str]
    is_valid: bool
    score: float
    departure_ts: Optional[int] = None
    duration_min: Optional[float] = None
    path: Optional[List[str]] = None
    explored_edges: Optional[List[Tuple[str, str]]] = None
    corrections: Optional[List] = None


class TravelResolver:
    """Resolve free-text travel orders into station ids with graph validation."""

    def __init__(
        self,
        stations_path: str | None = None,
        connections_path: str | None = None,
        schedule_path: str | None = None,
        k_neighbors: int = DEFAULT_K_NEIGHBORS,
        cache: Cache | None = None,
        logger=None,
        phonetic_corrector=None,
    ) -> None:
        # Start HF model loading immediately in background thread
        self.hf_extractor = HFExtractor()
        self.hf_extractor.start_preload()

        self.stations_path = Path(stations_path) if stations_path else DEFAULT_DATASET_PATH
        self.connections_path = Path(connections_path) if connections_path else DEFAULT_CONNECTIONS_PATH
        self.schedule_path = Path(schedule_path) if schedule_path else None
        self.stations = pl.read_parquet(self.stations_path)
        self.graph = build_graph(
            stations_path=self.stations_path,
            connections_path=self.connections_path,
            schedule_path=self.schedule_path,
            k_neighbors=k_neighbors,
        )
        self.cache = cache
        self.logger = logger
        self.phonetic_corrector = phonetic_corrector
        self.aliases = self._build_aliases(self.stations)
        self.alias_best = self._build_alias_best(self.aliases)
        self.id_to_name = {row["station_id"]: row["name"] for row in self.stations.to_dicts()}
        self._interpolated_ids = frozenset(
            row["station_id"] for row in self.stations.to_dicts()
            if row.get("passengers") == "I"
        )
        self.transfer_lookup = self._load_connection_rules(DEFAULT_CONNECTION_RULES_PATH)
        self.spacy_extractor = SpacyExtractor()

    @staticmethod
    def _build_aliases(stations: pl.DataFrame) -> List[Tuple[str, str]]:
        """Return list of (station_id, alias_norm).

        Skip city-based aliases for interpolated stations (passengers='I')
        to avoid polluting the matching with approximate data.
        """
        aliases: List[Tuple[str, str]] = []
        for row in stations.to_dicts():
            sid = row["station_id"]
            name_norm = row["name_norm"]
            city_norm = normalize_name(row["city"])
            is_interpolated = row.get("passengers") == "I"
            # Interpolated stations only get their own name as alias
            if is_interpolated:
                if name_norm and len(name_norm) >= 3:
                    aliases.append((sid, name_norm))
                continue
            for alias in [
                name_norm,
                f"{name_norm} {city_norm}".strip(),
                f"{city_norm} {name_norm}".strip(),
                city_norm,
            ]:
                if alias and len(alias) >= 3:
                    aliases.append((sid, alias))
        # de-duplicate while preserving order
        seen = set()
        unique: List[Tuple[str, str]] = []
        for sid, alias in aliases:
            key = (sid, alias)
            if key in seen or not alias:
                continue
            seen.add(key)
            unique.append((sid, alias))
        return unique

    def _build_alias_best(self, aliases: List[Tuple[str, str]]) -> Dict[str, str]:
        """Choose best station per alias.

        Prefer confirmed passenger stations (O/G) over interpolated (I),
        then by schedule service volume (total departures across all edges).
        Service volume is more reliable than graph degree because degree is
        inflated by KNN edges on geographically central but minor stations.
        """
        station_quality = {
            row["station_id"]: row.get("passengers", "O")
            for row in self.stations.to_dicts()
        }
        quality_score = {"O": 2, "G": 1, "I": 0}
        # Compute service volume: total scheduled departures per station
        service_vol: Dict[str, int] = {}
        for node in self.graph.nodes:
            total = 0
            for _, _, data in self.graph.edges(node, data=True):
                total += len(data.get("departures_ts", []))
            service_vol[node] = total
        best: Dict[str, Tuple[int, int, str]] = {}
        for sid, alias in aliases:
            q = quality_score.get(station_quality.get(sid, "O"), 2)
            vol = service_vol.get(sid, 0)
            current = best.get(alias)
            if current is None or (q, vol) > (current[0], current[1]):
                best[alias] = (q, vol, sid)
        return {alias: sid for alias, (q, vol, sid) in best.items()}

    @staticmethod
    def _load_connection_rules(path: Path) -> Dict[Tuple[str, str], int]:
        """
        Load transfer buffer rules (seconds) keyed by (arr_uic, dep_uic).
        Columns expected: ARRIVAL_STATION_UIC, DEPARTURE_STATION_UIC, MIN_DELAY (minutes).
        """
        lookup: Dict[Tuple[str, str], int] = {}
        if not Path(path).exists():
            return lookup
        try:
            frame = pl.read_csv(path)
            required = {"ARRIVAL_STATION_UIC", "DEPARTURE_STATION_UIC", "MIN_DELAY"}
            if not required.issubset(set(frame.columns)):
                return lookup
            for row in frame.to_dicts():
                arr = str(row["ARRIVAL_STATION_UIC"])
                dep = str(row["DEPARTURE_STATION_UIC"])
                delay_min = int(row["MIN_DELAY"])
                lookup[(arr, dep)] = delay_min * 60
        except Exception:
            return lookup
        return lookup

    def get_transfer_buffer(self, arrival_uic: str, departure_uic: str) -> int:
        """Return transfer buffer in seconds."""
        return self.transfer_lookup.get((arrival_uic, departure_uic), 0)

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _city_alternatives(self, station_id: str) -> List[str]:
        """Return terminus stations in the same city cluster.

        1. BFS over is_transfer edges to discover the city cluster (graph-based).
        2. Filter: exclude interpolated stations.
        3. Filter: only keep stations whose name contains the city name
           (e.g. "Marseille-St-Charles" for city Marseille, not "L'Estaque").
        """
        if station_id not in self.graph:
            return [station_id]
        # Get city name for this station
        city_norm = self.graph.nodes[station_id].get("name_norm", "")
        city_name = normalize_name(self.graph.nodes[station_id].get("city", ""))
        if not city_name:
            return [station_id]
        # BFS on transfer edges
        visited: set[str] = set()
        queue = [station_id]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self.graph.neighbors(node):
                if self.graph[node][neighbor].get("is_transfer"):
                    queue.append(neighbor)
        # Keep terminus stations: non-interpolated + name contains city
        alts = [
            sid for sid in visited
            if sid not in self._interpolated_ids
            and city_name in (self.graph.nodes[sid].get("name_norm") or "")
        ]
        return alts if alts else [station_id]

    def _is_specific_station(self, fragment: str | None, station_id: str | None) -> bool:
        """Return True if the fragment refers to a specific station, not just a city."""
        if not fragment or not station_id or station_id not in self.graph:
            return False
        frag_norm = normalize_name(fragment)
        city_norm = normalize_name(self.graph.nodes[station_id].get("city", ""))
        # If the fragment is more specific than just the city name, it's a
        # specific station choice that should be respected.
        return frag_norm != city_norm

    def _best_station(self, fragment: str) -> Tuple[Optional[str], float]:
        """Find best matching station id for fragment."""
        fragment = fragment.strip()
        if not fragment:
            return None, 0.0
        # Normalize the fragment for matching against normalized aliases
        fragment_norm = normalize_name(fragment)
        if not fragment_norm:
            return None, 0.0
        # Direct station_id match
        if fragment_norm in self.id_to_name:
            return fragment_norm, 1.0
        # Direct alias best match (prefer main station in city)
        if fragment_norm in self.alias_best:
            return self.alias_best[fragment_norm], 0.95
        best_sid: Optional[str] = None
        best_score = 0.0
        # Fast path: substring hits
        for sid, alias in self.aliases:
            if alias and alias in fragment_norm:
                score = len(alias) / len(fragment_norm)
                if score > best_score:
                    # Prefer main station via alias_best
                    best_sid = self.alias_best.get(alias, sid)
                    best_score = score
        # Fallback approximate
        if best_score < 0.6:
            for sid, alias in self.aliases:
                score = self._similarity(fragment_norm, alias)
                if score > best_score:
                    best_sid = self.alias_best.get(alias, sid)
                    best_score = score
        return best_sid, best_score

    @staticmethod
    def _strip_via(text_norm: str) -> Tuple[str, Optional[str]]:
        via_patterns = [
            r"\bvia\s+(?P<via>[a-z0-9 ']+)",
            r"\ben passant par\s+(?P<via>[a-z0-9 ']+)",
            r"\bpar\s+(?P<via>[a-z0-9 ']+)",
        ]
        for pat in via_patterns:
            m = re.search(pat, text_norm)
            if m:
                via = m.group("via").strip()
                text_norm = text_norm.replace(m.group(0), " ")
                return text_norm, via
        return text_norm, None

    def _extract_segments(self, text_norm: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        text_norm = " ".join(text_norm.split())
        text_norm, via_fragment = self._strip_via(text_norm)

        patterns = [
            r"(?:de|depuis)\s+(?P<from>.+?)\s+(?:a|à|vers|pour|jusqu a)\s+(?P<to>.+)",
            r"(?:a|à|vers|pour)\s+(?P<to>.+?)\s+(?:depuis)\s+(?P<from>.+)",
            r"(?P<from>\b[\w ']+)\s+(?:vers|->|jusqu a)\s+(?P<to>[\w ']+)",
            r"(?:de|depuis)\s+(?P<from>.+?)\s+(?P<to>[^,]+?)(?:\s+(?:le|apres|après|a|à|vers|pour|jusqu|via|par)\b|$)",
        ]
        for pat in patterns:
            m = re.search(pat, text_norm)
            if m:
                return m.group("from").strip(), m.group("to").strip(), via_fragment

        # Fallback: find first two alias occurrences, preferring main stations
        hits: List[Tuple[int, str]] = []
        seen_alias: set[str] = set()
        for sid, alias in self.aliases:
            if alias in seen_alias:
                continue
            m = re.search(rf"\b{re.escape(alias)}\b", text_norm)
            pos = m.start() if m else -1
            if pos != -1:
                # Use alias_best to prefer main station (highest degree)
                best_sid = self.alias_best.get(alias, sid)
                hits.append((pos, best_sid))
                seen_alias.add(alias)
        hits.sort(key=lambda x: x[0])
        if len(hits) >= 2:
            return hits[0][1], hits[1][1], via_fragment

        return None, None, via_fragment

    def resolve_order(self, sentence_id: str, text: str, target_ts: int | None = None) -> TravelOrder:
        """Parse a sentence, match stations, validate on graph, return decision."""
        norm = normalize_name(text)

        # cache
        if self.cache:
            cached = self.cache.get(norm)
            if cached:
                dep_id, arr_id, score_str = cached.split("|")
                return TravelOrder(sentence_id, dep_id or None, arr_id or None, [], True, float(score_str))

        from_frag, to_frag, via_frag = self._extract_segments(norm)
        # Strip trailing date expressions that pollute station matching
        _date_tail = re.compile(
            r"\s+(?:le\s+)?\d{1,2}\s+(?:janvier|fevrier|mars|avril|mai|juin|"
            r"juillet|aout|septembre|octobre|novembre|decembre)\b.*$"
        )
        if from_frag:
            from_frag = _date_tail.sub("", from_frag).strip() or from_frag
        if to_frag:
            to_frag = _date_tail.sub("", to_frag).strip() or to_frag
        extracted_dates: List[str] = []
        # HF extractor: only override regex when it produces a better station match
        if self.hf_extractor.is_ready():
            spans = self.hf_extractor.extract(text)
            if spans.depart and spans.depart.strip():
                _, hf_score = self._best_station(spans.depart)
                _, cur_score = self._best_station(from_frag or "")
                if hf_score >= cur_score:
                    from_frag = spans.depart
            if spans.arrivee and spans.arrivee.strip():
                _, hf_score = self._best_station(spans.arrivee)
                _, cur_score = self._best_station(to_frag or "")
                if hf_score >= cur_score:
                    to_frag = spans.arrivee
            if spans.vias:
                via_frag = spans.vias[0]
            extracted_dates = spans.dates
        # Try spaCy if still missing
        if self.spacy_extractor.is_ready() and (from_frag is None or to_frag is None):
            s_from, s_to, _ = self.spacy_extractor.extract(text)
            if from_frag is None and s_from:
                from_frag = s_from
            if to_frag is None and s_to:
                to_frag = s_to

        # Apply phonetic correction only to station fragments (not dates/other text)
        corrections = []
        if self.phonetic_corrector:
            for frag, label in [(from_frag, "from"), (to_frag, "to"), (via_frag, "via")]:
                if not frag:
                    continue
                cr = self.phonetic_corrector.correct_fragment(frag)
                if cr.corrections:
                    if label == "from":
                        from_frag = cr.corrected_text
                    elif label == "to":
                        to_frag = cr.corrected_text
                    else:
                        via_frag = cr.corrected_text
                    corrections.extend(cr.corrections)

        dep_id, dep_score = self._best_station(from_frag or "")
        arr_id, arr_score = self._best_station(to_frag or "")
        via_id, via_score = (None, 0.0)
        if via_frag:
            via_id, via_score = self._best_station(via_frag)

        scores = [s for s in [dep_score, arr_score, via_score] if s]
        overall_score = min(scores) if scores else 0.0

        valid = dep_id is not None and arr_id is not None and overall_score >= 0.5
        via_ids: List[str] = [via_id] if via_id else []

        departure_ts: Optional[int] = None
        total_duration: Optional[int] = None

        # If we extracted date spans from HF, try to parse them into a target_ts if none provided
        if extracted_dates and target_ts is None:
            from main import parse_datetime_from_text

            for dt_text in extracted_dates:
                parsed_ts = parse_datetime_from_text(dt_text)
                if parsed_ts:
                    target_ts = parsed_ts
                    break

        best_path: Optional[List[str]] = None
        best_explored: Optional[List[Tuple[str, str]]] = None

        if valid:
            # Use city alternatives only when the user gave a city name, not
            # a specific station.  E.g. "Lyon" → try all Lyon stations, but
            # "Lyon Part-Dieu" → use Part-Dieu only.
            dep_alts = (
                [dep_id] if self._is_specific_station(from_frag, dep_id)
                else self._city_alternatives(dep_id)
            )
            arr_alts = (
                [arr_id] if self._is_specific_station(to_frag, arr_id)
                else self._city_alternatives(arr_id)
            )

            best_route = None  # (dep_id, arr_id, departure_ts, total_duration, path)

            for d in dep_alts:
                for a in arr_alts:
                    if d == a:
                        continue
                    try:
                        if target_ts is not None:
                            path, d_ts, a_ts = compute_earliest_route(
                                self.graph,
                                d,
                                a,
                                departure_ts=target_ts,
                                buffer_fn=self.get_transfer_buffer,
                            )
                            dur = a_ts - d_ts
                            if best_route is None or dur < best_route[3]:
                                best_route = (d, a, d_ts, dur, path)
                        else:
                            path, explored = compute_route_with_exploration(
                                self.graph, d, a,
                            )
                            weight_min = sum(
                                self.graph[path[i]][path[i + 1]].get("weight", 1)
                                for i in range(len(path) - 1)
                            )
                            dur = weight_min * 60  # minutes → seconds
                            if best_route is None or dur < best_route[3]:
                                best_route = (d, a, None, dur, path)
                                best_explored = explored
                    except ValueError:
                        continue

            if best_route is not None:
                dep_id, arr_id = best_route[0], best_route[1]
                departure_ts = best_route[2]
                total_duration = best_route[3]  # always seconds
                best_path = best_route[4]
            else:
                valid = False

        if self.cache and valid:
            self.cache.set(norm, f"{dep_id}|{arr_id}|{overall_score:.3f}")

        if self.logger:
            decision = {"depart_id": dep_id, "arrivee_id": arr_id, "via_ids": via_ids}
            if departure_ts is not None:
                decision["departure_ts"] = departure_ts
            if total_duration:
                decision["duration_min"] = total_duration / 60
            log_decision(self.logger, sentence_id=sentence_id, decision=decision, score=overall_score, latency_ms=0.0)

        return TravelOrder(
            sentence_id=sentence_id,
            departure_id=dep_id if valid else None,
            arrival_id=arr_id if valid else None,
            via_ids=via_ids if valid else [],
            is_valid=valid,
            score=overall_score,
            departure_ts=departure_ts if valid else None,
            duration_min=(total_duration / 60) if valid and total_duration else None,
            path=[s for s in best_path if not self.id_to_name.get(s, "").startswith("Halte-")] if valid and best_path else None,
            explored_edges=best_explored if valid else None,
            corrections=corrections or None,
        )


@lru_cache(maxsize=1)
def default_resolver() -> TravelResolver:
    """Singleton resolver with default dataset and graph."""
    return TravelResolver()
