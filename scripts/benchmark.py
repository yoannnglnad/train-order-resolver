"""Benchmark script for Travel Order Resolver performance."""
import json
import os
import time
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def run_benchmark(label="benchmark"):
    results = {"label": label, "timestamp": time.time()}

    # 1. Import time
    t0 = time.perf_counter()
    from src.nlp.inference import TravelResolver
    from main import parse_datetime_from_text
    results["import_time_s"] = time.perf_counter() - t0

    # 2. Init time
    t1 = time.perf_counter()
    resolver = TravelResolver()
    results["init_time_s"] = time.perf_counter() - t1

    # 3. Query times
    phrases = [
        "Je veux aller de Bordeaux à Strasbourg le 20 mars",
        "Trajet Rouen vers Lyon le 15 avril",
        "Nantes vers Lille le 10 mai",
        "Marseille vers Rennes le 22 mars",
        "Dijon vers Bordeaux le 12 mai",
    ]

    query_times = []
    query_results = []
    for i, phrase in enumerate(phrases):
        t = time.perf_counter()
        target_ts = parse_datetime_from_text(phrase)
        order = resolver.resolve_order(str(i+1), phrase, target_ts=target_ts)
        elapsed = time.perf_counter() - t
        query_times.append(elapsed)
        dep = resolver.id_to_name.get(order.departure_id, 'INVALID')
        arr = resolver.id_to_name.get(order.arrival_id, 'INVALID')
        dur = f"{order.duration_min:.0f}min" if order.duration_min else "N/A"
        query_results.append({"phrase": phrase, "dep": dep, "arr": arr, "duration": dur, "time_ms": elapsed*1000})

    results["queries"] = query_results
    results["query_avg_ms"] = sum(query_times) / len(query_times) * 1000
    results["query_p95_ms"] = sorted(query_times)[int(0.95 * len(query_times))] * 1000
    results["total_time_s"] = results["import_time_s"] + results["init_time_s"] + sum(query_times)

    # Memory
    import resource
    results["memory_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    return results

if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "benchmark"
    # Only clear caches if "cold" is in the label
    if "cold" in label:
        os.system("rm -f data/cache.sqlite data/cache/graph.pickle")

    results = run_benchmark(label)

    print(f"\n{'='*50}")
    print(f"BENCHMARK: {label}")
    print(f"{'='*50}")
    print(f"Import time:    {results['import_time_s']:.2f}s")
    print(f"Init time:      {results['init_time_s']:.2f}s")
    print(f"Query avg:      {results['query_avg_ms']:.0f}ms")
    print(f"Query p95:      {results['query_p95_ms']:.0f}ms")
    print(f"Total time:     {results['total_time_s']:.2f}s")
    print(f"Memory:         {results['memory_mb']:.0f}MB")
    print(f"\nQueries:")
    for q in results["queries"]:
        print(f"  {q['dep']:25s} -> {q['arr']:20s} {q['duration']:>8s}  ({q['time_ms']:.0f}ms)")

    # Save
    out_path = Path(f"data/logs/{label}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
