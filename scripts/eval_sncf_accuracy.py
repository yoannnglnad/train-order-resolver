"""Evaluate pipeline accuracy against SNCF/Navitia API as ground truth."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import median

import requests
from dotenv import load_dotenv

# ── ensure project root on sys.path ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from scripts.eval_dataset import (
    EVAL_QUERIES,
    EVAL_DATETIME,
    EVAL_TIMESTAMP,
    CITY_SEARCH_OVERRIDES,
)
from scripts.compare_sncf_api import sncf_search_stop
from src.nlp.inference import TravelResolver
from src.pathfinding.prepare_stations import normalize_name

API_KEY = os.environ["SNCF_API_KEY"]
BASE_URL = os.environ.get("SNCF_API_BASE_URL", "https://api.sncf.com/v1")


# ── Helpers ──


def extract_uic(stop_area_id: str) -> str | None:
    """Extract UIC code from Navitia stop_area id.

    Example: 'stop_area:SNCF:87686006' -> '87686006'
    """
    parts = stop_area_id.split(":")
    for part in parts:
        if part.isdigit() and len(part) == 8:
            return part
    return None


def sncf_journeys_safe(from_id: str, to_id: str, dt: str, count: int = 5) -> list[dict]:
    """Fetch journeys from SNCF API, returning [] on 404 (date out of bounds)."""
    r = requests.get(
        f"{BASE_URL}/coverage/sncf/journeys",
        params={"from": from_id, "to": to_id, "datetime": dt, "count": count},
        auth=(API_KEY, ""),
        timeout=15,
    )
    if r.status_code == 404:
        return []
    r.raise_for_status()
    return r.json().get("journeys", [])


def search_city_stop(city: str) -> dict | None:
    """Search for a city's main station, using overrides for ambiguous names."""
    search_term = CITY_SEARCH_OVERRIDES.get(city, city)
    return sncf_search_stop(search_term)


def duration_bucket(pipeline_min: float, api_min: float) -> str:
    """Classify duration deviation: GREEN < 10%, ORANGE 10-20%, RED > 20%."""
    if api_min == 0:
        return "RED"
    deviation = abs(pipeline_min - api_min) / api_min
    if deviation < 0.10:
        return "GREEN"
    elif deviation < 0.20:
        return "ORANGE"
    return "RED"


# ── Step 1: Collect API ground truth ──


def collect_api_ground_truth(queries: list[dict]) -> list[dict]:
    """For each query, call SNCF API to get ground truth stations + shortest duration."""
    results = []
    for q in queries:
        entry = {
            "id": q["id"],
            "category": q["category"],
            "phrase": q["phrase"],
            "api_found": False,
            "api_dep_uic": None,
            "api_arr_uic": None,
            "api_dep_name": None,
            "api_arr_name": None,
            "api_duration_min": None,
        }

        try:
            dep_place = search_city_stop(q["expected_dep_city"])
            time.sleep(0.5)
            arr_place = search_city_stop(q["expected_arr_city"])
            time.sleep(0.5)

            if not dep_place or not arr_place:
                print(f"  [{q['id']}] API: station not found "
                      f"(dep={q['expected_dep_city']}, arr={q['expected_arr_city']})")
                results.append(entry)
                continue

            dep_id = dep_place["id"]
            arr_id = arr_place["id"]
            entry["api_dep_name"] = dep_place.get("name", dep_id)
            entry["api_arr_name"] = arr_place.get("name", arr_id)
            entry["api_dep_uic"] = extract_uic(dep_id)
            entry["api_arr_uic"] = extract_uic(arr_id)

            journeys = sncf_journeys_safe(dep_id, arr_id, EVAL_DATETIME, count=5)
            time.sleep(0.5)

            if journeys:
                shortest = min(journeys, key=lambda j: j["duration"])
                entry["api_duration_min"] = shortest["duration"] / 60
                entry["api_found"] = True
                print(f"  [{q['id']}] API: {entry['api_dep_name']} -> "
                      f"{entry['api_arr_name']} ({entry['api_duration_min']:.0f} min)")
            else:
                entry["api_found"] = True
                print(f"  [{q['id']}] API: {entry['api_dep_name']} -> "
                      f"{entry['api_arr_name']} (no journey)")

        except Exception as e:
            print(f"  [{q['id']}] API error: {e}")

        results.append(entry)

    return results


# ── Step 2: Run pipeline ──


def run_pipeline(queries: list[dict], resolver: TravelResolver) -> list[dict]:
    """Run our pipeline on each query and collect results."""
    # Build station_id -> city lookup
    city_lookup = {
        row["station_id"]: row["city"]
        for row in resolver.stations.to_dicts()
    }

    results = []
    for q in queries:
        entry = {
            "id": q["id"],
            "pipeline_valid": False,
            "pipeline_dep_id": None,
            "pipeline_arr_id": None,
            "pipeline_dep_name": None,
            "pipeline_arr_name": None,
            "pipeline_dep_city": None,
            "pipeline_arr_city": None,
            "pipeline_duration_min": None,
        }

        try:
            order = resolver.resolve_order(q["id"], q["phrase"], target_ts=EVAL_TIMESTAMP)
            if order.is_valid:
                entry["pipeline_valid"] = True
                entry["pipeline_dep_id"] = order.departure_id
                entry["pipeline_arr_id"] = order.arrival_id
                entry["pipeline_dep_name"] = resolver.id_to_name.get(order.departure_id, order.departure_id)
                entry["pipeline_arr_name"] = resolver.id_to_name.get(order.arrival_id, order.arrival_id)
                entry["pipeline_dep_city"] = city_lookup.get(order.departure_id)
                entry["pipeline_arr_city"] = city_lookup.get(order.arrival_id)
                entry["pipeline_duration_min"] = order.duration_min
        except Exception as e:
            print(f"  [{q['id']}] Pipeline error: {e}")

        results.append(entry)

    return results


# ── Step 3: Compare ──


def compare_results(
    queries: list[dict],
    api_results: list[dict],
    pipeline_results: list[dict],
) -> list[dict]:
    """Merge API and pipeline results, compute per-query comparison."""
    comparisons = []

    for q, api, pipe in zip(queries, api_results, pipeline_results):
        comp = {
            "id": q["id"],
            "category": q["category"],
            "phrase": q["phrase"],
            # API
            "api_found": api["api_found"],
            "api_dep_uic": api["api_dep_uic"],
            "api_arr_uic": api["api_arr_uic"],
            "api_dep_name": api["api_dep_name"],
            "api_arr_name": api["api_arr_name"],
            "api_duration_min": api["api_duration_min"],
            # Pipeline
            "pipeline_valid": pipe["pipeline_valid"],
            "pipeline_dep_id": pipe["pipeline_dep_id"],
            "pipeline_arr_id": pipe["pipeline_arr_id"],
            "pipeline_dep_name": pipe["pipeline_dep_name"],
            "pipeline_arr_name": pipe["pipeline_arr_name"],
            "pipeline_dep_city": pipe["pipeline_dep_city"],
            "pipeline_arr_city": pipe["pipeline_arr_city"],
            "pipeline_duration_min": pipe["pipeline_duration_min"],
            # Comparison
            "dep_match": False,
            "arr_match": False,
            "both_match": False,
            "duration_bucket": None,
            "duration_diff_min": None,
            "duration_deviation_pct": None,
        }

        # Station matching — compare at city level using accent-normalized names.
        # Multi-station cities (Paris, Lyon, etc.) can legitimately resolve to
        # different UIC codes; what matters is the correct city.
        if api["api_found"] and pipe["pipeline_valid"]:
            expected_dep = normalize_name(q["expected_dep_city"])
            expected_arr = normalize_name(q["expected_arr_city"])
            pipe_dep_city = normalize_name(pipe["pipeline_dep_city"] or "")
            pipe_arr_city = normalize_name(pipe["pipeline_arr_city"] or "")

            comp["dep_match"] = pipe_dep_city == expected_dep
            comp["arr_match"] = pipe_arr_city == expected_arr
            comp["both_match"] = comp["dep_match"] and comp["arr_match"]

        # Duration comparison
        if (
            api["api_found"]
            and pipe["pipeline_valid"]
            and api["api_duration_min"] is not None
            and pipe["pipeline_duration_min"] is not None
        ):
            diff = pipe["pipeline_duration_min"] - api["api_duration_min"]
            comp["duration_diff_min"] = round(diff, 1)
            if api["api_duration_min"] > 0:
                comp["duration_deviation_pct"] = round(
                    abs(diff) / api["api_duration_min"] * 100, 1
                )
            comp["duration_bucket"] = duration_bucket(
                pipe["pipeline_duration_min"], api["api_duration_min"]
            )

        comparisons.append(comp)

    return comparisons


# ── Step 4: Aggregate and report ──


def compute_aggregates(comparisons: list[dict]) -> dict:
    """Compute summary metrics from comparison results."""
    total = len(comparisons)
    comparable = [c for c in comparisons if c["api_found"] and c["pipeline_valid"]]
    n_comparable = len(comparable)

    # Station matching
    dep_matches = sum(1 for c in comparable if c["dep_match"])
    arr_matches = sum(1 for c in comparable if c["arr_match"])
    both_matches = sum(1 for c in comparable if c["both_match"])

    # Duration metrics (only where both have durations)
    with_duration = [c for c in comparable if c["duration_diff_min"] is not None]
    n_with_dur = len(with_duration)

    abs_errors = [abs(c["duration_diff_min"]) for c in with_duration]
    deviations = [c["duration_deviation_pct"] for c in with_duration if c["duration_deviation_pct"] is not None]

    green = sum(1 for c in with_duration if c["duration_bucket"] == "GREEN")
    orange = sum(1 for c in with_duration if c["duration_bucket"] == "ORANGE")
    red = sum(1 for c in with_duration if c["duration_bucket"] == "RED")

    # Category breakdown
    categories = sorted(set(c["category"] for c in comparisons))
    breakdown = {}
    for cat in categories:
        cat_items = [c for c in comparable if c["category"] == cat]
        cat_dur = [c for c in cat_items if c["duration_diff_min"] is not None]
        breakdown[cat] = {
            "total": sum(1 for c in comparisons if c["category"] == cat),
            "comparable": len(cat_items),
            "dep_match": sum(1 for c in cat_items if c["dep_match"]),
            "arr_match": sum(1 for c in cat_items if c["arr_match"]),
            "both_match": sum(1 for c in cat_items if c["both_match"]),
            "green": sum(1 for c in cat_dur if c["duration_bucket"] == "GREEN"),
            "orange": sum(1 for c in cat_dur if c["duration_bucket"] == "ORANGE"),
            "red": sum(1 for c in cat_dur if c["duration_bucket"] == "RED"),
            "mae_min": round(sum(abs(c["duration_diff_min"]) for c in cat_dur) / len(cat_dur), 1) if cat_dur else None,
        }

    return {
        "total_queries": total,
        "comparable": n_comparable,
        "api_failures": sum(1 for c in comparisons if not c["api_found"]),
        "pipeline_failures": sum(1 for c in comparisons if not c["pipeline_valid"]),
        "station_matching": {
            "dep_match": dep_matches,
            "arr_match": arr_matches,
            "both_match": both_matches,
            "dep_match_rate": round(dep_matches / n_comparable, 3) if n_comparable else 0,
            "arr_match_rate": round(arr_matches / n_comparable, 3) if n_comparable else 0,
            "both_match_rate": round(both_matches / n_comparable, 3) if n_comparable else 0,
        },
        "duration": {
            "with_duration": n_with_dur,
            "green": green,
            "orange": orange,
            "red": red,
            "green_pct": round(green / n_with_dur * 100, 1) if n_with_dur else 0,
            "orange_pct": round(orange / n_with_dur * 100, 1) if n_with_dur else 0,
            "red_pct": round(red / n_with_dur * 100, 1) if n_with_dur else 0,
            "mae_min": round(sum(abs_errors) / n_with_dur, 1) if n_with_dur else None,
            "median_ae_min": round(median(abs_errors), 1) if abs_errors else None,
            "mean_deviation_pct": round(sum(deviations) / len(deviations), 1) if deviations else None,
        },
        "category_breakdown": breakdown,
    }


def print_report(comparisons: list[dict], aggregates: dict) -> None:
    """Print a formatted console report."""
    print("\n" + "=" * 90)
    print("  EVALUATION: Pipeline vs SNCF API")
    print("=" * 90)

    # Per-query detail table
    print(f"\n{'ID':<12} {'Dep Match':>10} {'Arr Match':>10} {'API dur':>10} {'Pipe dur':>10} {'Diff':>8} {'Bucket':>8}")
    print("-" * 70)
    for c in comparisons:
        dep_m = "OK" if c["dep_match"] else ("MISS" if c["api_found"] and c["pipeline_valid"] else "N/A")
        arr_m = "OK" if c["arr_match"] else ("MISS" if c["api_found"] and c["pipeline_valid"] else "N/A")
        api_d = f"{c['api_duration_min']:.0f}" if c["api_duration_min"] is not None else "N/A"
        pipe_d = f"{c['pipeline_duration_min']:.0f}" if c["pipeline_duration_min"] is not None else "N/A"
        diff = f"{c['duration_diff_min']:+.0f}" if c["duration_diff_min"] is not None else "N/A"
        bucket = c["duration_bucket"] or "N/A"
        print(f"{c['id']:<12} {dep_m:>10} {arr_m:>10} {api_d:>10} {pipe_d:>10} {diff:>8} {bucket:>8}")

    # Summary
    sm = aggregates["station_matching"]
    dur = aggregates["duration"]

    print(f"\n{'─' * 50}")
    print(f"  SUMMARY ({aggregates['comparable']}/{aggregates['total_queries']} comparable)")
    print(f"{'─' * 50}")
    print(f"  API failures     : {aggregates['api_failures']}")
    print(f"  Pipeline failures: {aggregates['pipeline_failures']}")

    print(f"\n  Station Matching:")
    print(f"    Departure match : {sm['dep_match']}/{aggregates['comparable']} ({sm['dep_match_rate']:.1%})")
    print(f"    Arrival match   : {sm['arr_match']}/{aggregates['comparable']} ({sm['arr_match_rate']:.1%})")
    print(f"    Both match      : {sm['both_match']}/{aggregates['comparable']} ({sm['both_match_rate']:.1%})")

    print(f"\n  Duration Accuracy ({dur['with_duration']} with durations):")
    print(f"    GREEN  (< 10%) : {dur['green']:>3}  ({dur['green_pct']:.1f}%)")
    print(f"    ORANGE (10-20%): {dur['orange']:>3}  ({dur['orange_pct']:.1f}%)")
    print(f"    RED    (> 20%) : {dur['red']:>3}  ({dur['red_pct']:.1f}%)")
    print(f"    MAE            : {dur['mae_min']} min" if dur["mae_min"] is not None else "    MAE            : N/A")
    print(f"    Median AE      : {dur['median_ae_min']} min" if dur["median_ae_min"] is not None else "    Median AE      : N/A")
    print(f"    Mean deviation : {dur['mean_deviation_pct']}%" if dur["mean_deviation_pct"] is not None else "    Mean deviation : N/A")

    # Category breakdown
    print(f"\n  Category Breakdown:")
    print(f"  {'Category':<20} {'Both':>6} {'Green':>6} {'Org':>6} {'Red':>6} {'MAE':>8}")
    print(f"  {'─' * 54}")
    for cat, data in aggregates["category_breakdown"].items():
        both = f"{data['both_match']}/{data['comparable']}" if data["comparable"] else "N/A"
        g = str(data["green"])
        o = str(data["orange"])
        r = str(data["red"])
        mae = f"{data['mae_min']}m" if data["mae_min"] is not None else "N/A"
        print(f"  {cat:<20} {both:>6} {g:>6} {o:>6} {r:>6} {mae:>8}")

    print()


# ── Main ──


def main() -> None:
    print("=" * 90)
    print("  Pipeline vs SNCF API Accuracy Evaluation")
    print(f"  Date: {EVAL_DATETIME}  |  Queries: {len(EVAL_QUERIES)}")
    print("=" * 90)

    # Step 1: API ground truth
    print("\n[1/4] Collecting SNCF API ground truth...")
    api_results = collect_api_ground_truth(EVAL_QUERIES)

    # Step 2: Pipeline
    print("\n[2/4] Running pipeline...")
    resolver = TravelResolver()
    pipeline_results = run_pipeline(EVAL_QUERIES, resolver)

    # Step 3: Compare
    print("\n[3/4] Comparing results...")
    comparisons = compare_results(EVAL_QUERIES, api_results, pipeline_results)

    # Step 4: Aggregate and report
    print("\n[4/4] Computing aggregates...")
    aggregates = compute_aggregates(comparisons)

    print_report(comparisons, aggregates)

    # Save JSON
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_datetime": EVAL_DATETIME,
        "eval_timestamp": EVAL_TIMESTAMP,
        "total_queries": len(EVAL_QUERIES),
        "aggregates": aggregates,
        "details": comparisons,
    }

    out_path = Path("data/logs/eval_sncf_accuracy.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
