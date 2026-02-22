"""Compare our pipeline output with the SNCF/Navitia API for the same journey."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import os

import requests
from dotenv import load_dotenv

# ── ensure project root on sys.path ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.nlp.inference import TravelResolver
from main import parse_datetime_from_text

# ── SNCF API config ──
API_KEY = os.environ["SNCF_API_KEY"]
BASE_URL = os.environ.get("SNCF_API_BASE_URL", "https://api.sncf.com/v1")


def sncf_search_stop(name: str) -> dict | None:
    """Search for a stop_area by name via SNCF API."""
    r = requests.get(
        f"{BASE_URL}/coverage/sncf/places",
        params={"q": name, "type[]": "stop_area", "count": 1},
        auth=(API_KEY, ""),
        timeout=10,
    )
    r.raise_for_status()
    places = r.json().get("places", [])
    if not places:
        return None
    return places[0]


def sncf_journeys(from_id: str, to_id: str, dt: str) -> list[dict]:
    """Fetch journeys from SNCF API."""
    r = requests.get(
        f"{BASE_URL}/coverage/sncf/journeys",
        params={
            "from": from_id,
            "to": to_id,
            "datetime": dt,
            "count": 3,
        },
        auth=(API_KEY, ""),
        timeout=15,
    )
    r.raise_for_status()
    return r.json().get("journeys", [])


def fmt_ts(ts: int | None) -> str:
    if ts is None:
        return "N/A"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def fmt_navitia_dt(s: str) -> str:
    """Parse Navitia datetime string 'YYYYMMDDTHHMMSS' to readable format."""
    try:
        dt = datetime.strptime(s, "%Y%m%dT%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return s


def main() -> None:
    # ── Test phrase ──
    phrase = "Je voudrais aller de Paris à Lyon demain à 8h00"

    # Allow override via CLI
    if len(sys.argv) > 1:
        phrase = " ".join(sys.argv[1:])

    print("=" * 70)
    print(f"  PHRASE : {phrase}")
    print("=" * 70)

    # ────────────────────────────────────────────────────
    # 1. Notre pipeline
    # ────────────────────────────────────────────────────
    print("\n▶ PIPELINE (notre modèle)")
    print("-" * 40)

    t0 = time.perf_counter()
    resolver = TravelResolver()
    init_ms = (time.perf_counter() - t0) * 1000

    target_ts = parse_datetime_from_text(phrase)
    t0 = time.perf_counter()
    order = resolver.resolve_order("test-1", phrase, target_ts=target_ts)
    resolve_ms = (time.perf_counter() - t0) * 1000

    if order.is_valid:
        dep_name = resolver.id_to_name.get(order.departure_id, order.departure_id)
        arr_name = resolver.id_to_name.get(order.arrival_id, order.arrival_id)
        print(f"  Départ    : {dep_name} ({order.departure_id})")
        print(f"  Arrivée   : {arr_name} ({order.arrival_id})")
        print(f"  Score     : {order.score:.2f}")
        print(f"  Heure dép : {fmt_ts(order.departure_ts)}")
        if order.duration_min is not None:
            dep_dt = datetime.fromtimestamp(order.departure_ts, tz=timezone.utc)
            arr_dt = dep_dt + timedelta(minutes=order.duration_min)
            print(f"  Heure arr : {arr_dt.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Durée     : {order.duration_min:.0f} min")
        else:
            print("  Durée     : N/A (pas de schedule)")
    else:
        dep_name = "INVALID"
        arr_name = "INVALID"
        print("  Résultat  : INVALID")

    print(f"  Latence   : {resolve_ms:.0f} ms (init: {init_ms:.0f} ms)")

    # ────────────────────────────────────────────────────
    # 2. API SNCF
    # ────────────────────────────────────────────────────
    print("\n▶ API SNCF (Navitia)")
    print("-" * 40)

    # Resolve station names via the API
    from_name = dep_name if order.is_valid else "Paris"
    to_name = arr_name if order.is_valid else "Lyon"

    t0 = time.perf_counter()
    from_place = sncf_search_stop(from_name)
    to_place = sncf_search_stop(to_name)
    search_ms = (time.perf_counter() - t0) * 1000

    if not from_place or not to_place:
        print(f"  Erreur : gare non trouvée (from={from_place}, to={to_place})")
        return

    from_id = from_place["id"]
    to_id = to_place["id"]
    from_label = from_place.get("name", from_id)
    to_label = to_place.get("name", to_id)

    print(f"  Départ    : {from_label} ({from_id})")
    print(f"  Arrivée   : {to_label} ({to_id})")

    # Build datetime for API
    if target_ts:
        api_dt = datetime.fromtimestamp(target_ts, tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    else:
        api_dt = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y%m%dT%H%M%S")

    t0 = time.perf_counter()
    journeys = sncf_journeys(from_id, to_id, api_dt)
    journey_ms = (time.perf_counter() - t0) * 1000

    if not journeys:
        print("  Aucun itinéraire trouvé")
    else:
        for i, j in enumerate(journeys):
            dep_dt = fmt_navitia_dt(j["departure_date_time"])
            arr_dt = fmt_navitia_dt(j["arrival_date_time"])
            dur_min = j["duration"] // 60
            transfers = j["nb_transfers"]
            co2 = j.get("co2_emission", {})
            co2_str = f'{co2["value"]:.0f} {co2["unit"]}' if co2.get("value") else "N/A"

            sections_pt = [s for s in j.get("sections", []) if s["type"] == "public_transport"]
            modes = []
            for s in sections_pt:
                info = s.get("display_informations", {})
                mode = info.get("commercial_mode", "")
                code = info.get("code", "")
                headsign = info.get("headsign", "")
                modes.append(f"{mode} {code} ({headsign})".strip())

            print(f"\n  --- Trajet {i + 1} ---")
            print(f"  Heure dép : {dep_dt}")
            print(f"  Heure arr : {arr_dt}")
            print(f"  Durée     : {dur_min} min")
            print(f"  Corresp.  : {transfers}")
            print(f"  CO2       : {co2_str}")
            if modes:
                print(f"  Trains    : {' → '.join(modes)}")

    print(f"  Latence   : {journey_ms:.0f} ms (recherche gares: {search_ms:.0f} ms)")

    # ────────────────────────────────────────────────────
    # 3. Comparaison
    # ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMPARAISON")
    print("=" * 70)

    print(f"\n  {'':30s} {'Pipeline':>15s}  {'API SNCF':>15s}")
    print(f"  {'─' * 30} {'─' * 15}  {'─' * 15}")
    print(f"  {'Gare de départ':30s} {dep_name:>15s}  {from_label:>15s}")
    print(f"  {'Gare d\'arrivée':30s} {arr_name:>15s}  {to_label:>15s}")

    if order.is_valid and order.departure_ts:
        pipe_dep = fmt_ts(order.departure_ts)
    else:
        pipe_dep = "N/A"
    api_dep = fmt_navitia_dt(journeys[0]["departure_date_time"]) if journeys else "N/A"
    print(f"  {'Heure de départ':30s} {pipe_dep:>15s}  {api_dep:>15s}")

    if order.is_valid and order.duration_min is not None:
        pipe_dur = f"{order.duration_min:.0f} min"
    else:
        pipe_dur = "N/A"
    api_dur = f"{journeys[0]['duration'] // 60} min" if journeys else "N/A"
    print(f"  {'Durée':30s} {pipe_dur:>15s}  {api_dur:>15s}")

    if order.is_valid and order.duration_min and journeys:
        diff = order.duration_min - (journeys[0]["duration"] / 60)
        print(f"\n  Écart de durée : {diff:+.0f} min (pipeline vs API)")

    print()


if __name__ == "__main__":
    main()
