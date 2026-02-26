"""CLI entrypoint for Travel Order Resolver."""

from __future__ import annotations

import argparse
import csv
import time
import sys
from pathlib import Path
from typing import Iterable, TextIO

from src.nlp.inference import TravelResolver
from src.utils.cache import Cache
from src.utils.config import (
    DEFAULT_CACHE_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_INPUT_PATH,
    DEFAULT_K_NEIGHBORS,
    DEFAULT_LOG_PATH,
)
from src.utils.logging import get_json_logger, log_metrics

# Simple French date/time parsing (day month [year] and hh[h]mm)
from datetime import datetime, timezone, timedelta
import re

MONTHS_FR = {
    "janvier": 1,
    "fevrier": 2,
    "février": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "août": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
    "décembre": 12,
}


def parse_datetime_from_text(text: str) -> int | None:
    """Extract a coarse timestamp (UTC) from a French natural phrase."""
    norm = text.lower()
    now = datetime.now(timezone.utc)

    base_date = None
    if "apres-demain" in norm or "après-demain" in norm:
        base_date = (now + timedelta(days=2)).date()
    elif "demain" in norm:
        base_date = (now + timedelta(days=1)).date()

    # day + month + optional year
    date_match = re.search(
        r"(?:le\s+)?(?P<day>\d{1,2})\s+(?P<month>[a-zéèêûôîäëïöü]+)(?:\s+(?P<year>\d{4}))?",
        norm,
    )
    if date_match:
        day = int(date_match.group("day"))
        month_name = date_match.group("month")
        month = MONTHS_FR.get(month_name)
        if not month:
            return None
        year = int(date_match.group("year")) if date_match.group("year") else now.year
        base_date = datetime(year, month, day, tzinfo=timezone.utc).date()

    if base_date is None:
        return None

    # time component (after/après HH[h][mm] or HH:MM)
    time_match = re.search(r"(?:apres|après|ap|a|à|vers)?\s*(?P<h>\d{1,2})h(?P<m>\d{2})?", norm)
    if not time_match:
        time_match = re.search(r"(?P<hh>\d{1,2}):(?P<mm>\d{2})", norm)
        if time_match:
            hour = int(time_match.group("hh"))
            minute = int(time_match.group("mm"))
        else:
            hour = 0
            minute = 0
    else:
        hour = int(time_match.group("h"))
        minute = int(time_match.group("m") or 0)

    try:
        dt = datetime(base_date.year, base_date.month, base_date.day, hour, minute, tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(dt.timestamp())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve natural language travel orders into rail itineraries."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to an input file (one CSV-formatted sentence per line). Defaults to stdin.",
    )
    input_group.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Path to an audio file (WAV/MP3/FLAC) to transcribe and resolve.",
    )
    parser.add_argument(
        "--no-phonetic-correction",
        action="store_true",
        help="Disable phonetic correction of station names after transcription.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the output CSV. Defaults to stdout.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to curated stations parquet.",
    )
    parser.add_argument(
        "--connections",
        type=Path,
        default=None,
        help="Optional connections CSV/Parquet with columns from_id,to_id[,weight].",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=DEFAULT_K_NEIGHBORS,
        help="Number of nearest neighbors for graph edges.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Path to SQLite cache file.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache usage.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to JSONL decision log file.",
    )
    parser.add_argument(
        "--no-log-stdout",
        action="store_true",
        help="Disable logging to stdout.",
    )
    return parser.parse_args()


def _open_input(path: Path | None) -> TextIO:
    if path is None:
        return sys.stdin
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path.open("r", encoding="utf-8")


def _open_output(path: Path | None) -> TextIO:
    return sys.stdout if path is None else path.open("w", encoding="utf-8", newline="")


def main() -> int:
    args = parse_args()
    cache = None if args.no_cache else Cache(args.cache_path)
    logger = get_json_logger(
        log_path=args.log_path,
        stream_to_stdout=not args.no_log_stdout,
    )
    resolver = TravelResolver(
        stations_path=args.dataset,
        connections_path=args.connections,
        k_neighbors=args.k_neighbors,
        cache=cache,
        logger=logger,
    )

    # --- Audio mode: transcribe, extract entities, then correct station names ---
    if args.audio:
        from src.stt.transcriber import Transcriber
        from src.stt.phonetic_corrector import PhoneticCorrector

        transcriber = Transcriber()
        result = transcriber.transcribe(args.audio)
        phrase = result.text
        sys.stderr.write(f"[STT] Transcription: {phrase}\n")

        # Pass corrector to resolver: phonetic correction is applied only
        # to extracted station entities, not the full text.
        if not args.no_phonetic_correction:
            resolver.phonetic_corrector = PhoneticCorrector()

        with _open_output(args.output) as outfile:
            writer = csv.writer(outfile)
            writer.writerow(
                ["sentenceID", "depart", "arrivee", "depart_horaire", "arrivee_horaire", "duree_min"]
            )
            target_ts = parse_datetime_from_text(phrase)
            order = resolver.resolve_order("audio_1", phrase, target_ts=target_ts)
            if order.corrections:
                for c in order.corrections:
                    sys.stderr.write(
                        f'[STT] Correction: "{c.original}" -> "{c.corrected}" '
                        f"(distance={c.ipa_distance:.3f})\n"
                    )
            if not order.is_valid:
                writer.writerow(["audio_1", "INVALID", "", "", "", ""])
            else:
                dep_name = resolver.id_to_name[order.departure_id]
                arr_name = resolver.id_to_name[order.arrival_id]
                dep_time = ""
                arr_time = ""
                duree = ""
                if order.departure_ts is not None:
                    dep_dt = datetime.fromtimestamp(order.departure_ts, tz=timezone.utc)
                    dep_time = dep_dt.strftime("%Y-%m-%d %H:%M")
                    if order.duration_min is not None:
                        arr_dt = dep_dt + timedelta(minutes=order.duration_min)
                        arr_time = arr_dt.strftime("%Y-%m-%d %H:%M")
                        duree = f"{order.duration_min:.0f}"
                writer.writerow(["audio_1", dep_name, arr_name, dep_time, arr_time, duree])
        return 0

    with _open_input(args.input) as infile, _open_output(args.output) as outfile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None or "sentenceID" not in reader.fieldnames or "phrase" not in reader.fieldnames:
            sys.stderr.write("Input CSV must have headers: sentenceID,phrase\n")
            return 1
        writer = csv.writer(outfile)
        writer.writerow(["sentenceID", "depart", "arrivee", "depart_horaire", "arrivee_horaire", "duree_min"])
        metrics_total = 0
        latencies: list[float] = []
        invalid = 0
        graph_valid = 0
        for row in reader:
            sentence_id = row.get("sentenceID", "").strip()
            phrase = row.get("phrase", "")
            start = time.perf_counter()
            target_ts = parse_datetime_from_text(phrase)
            order = resolver.resolve_order(sentence_id, phrase, target_ts=target_ts)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            metrics_total += 1

            if not order.is_valid:
                invalid += 1
                writer.writerow([sentence_id, "INVALID", "", "", "", ""])
            else:
                graph_valid += 1
                dep_name = resolver.id_to_name[order.departure_id]
                arr_name = resolver.id_to_name[order.arrival_id]
                dep_time = ""
                arr_time = ""
                duree = ""
                if order.departure_ts is not None:
                    dep_dt = datetime.fromtimestamp(order.departure_ts, tz=timezone.utc)
                    dep_time = dep_dt.strftime("%Y-%m-%d %H:%M")
                    if order.duration_min is not None:
                        arr_dt = dep_dt + timedelta(minutes=order.duration_min)
                        arr_time = arr_dt.strftime("%Y-%m-%d %H:%M")
                        duree = f"{order.duration_min:.0f}"
                writer.writerow([sentence_id, dep_name, arr_name, dep_time, arr_time, duree])

        if metrics_total:
            mean_latency = sum(latencies) / metrics_total
            p95 = sorted(latencies)[int(0.95 * len(latencies)) - 1] if latencies else 0.0
            metrics = {
                "total": metrics_total,
                "invalid_count": invalid,
                "graph_valid_count": graph_valid,
                "invalid_rate": invalid / metrics_total,
                "graph_valid_rate": graph_valid / metrics_total,
                "latency_ms_mean": mean_latency,
                "latency_ms_p95": p95,
            }
            log_metrics(logger, metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
