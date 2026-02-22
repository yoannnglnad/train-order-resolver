"""FastAPI server exposing the Travel Order Resolver via HTTP."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nlp.inference import TravelResolver
from src.utils.config import DEFAULT_PHONETIC_INDEX_PATH

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

app = FastAPI(title="Travel Order Resolver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded singletons
_resolver: TravelResolver | None = None
_transcriber = None
_corrector = None


def _get_resolver() -> TravelResolver:
    global _resolver
    if _resolver is None:
        _resolver = TravelResolver()
    return _resolver


def _get_transcriber():
    global _transcriber
    if _transcriber is None:
        from src.stt.transcriber import Transcriber

        _transcriber = Transcriber()
    return _transcriber


def _get_corrector():
    global _corrector
    if _corrector is None:
        from src.stt.phonetic_corrector import PhoneticCorrector

        _corrector = PhoneticCorrector()
    return _corrector


def _parse_datetime_from_text(text: str) -> int | None:
    """Minimal French date parser (same logic as main.py)."""
    import re

    MONTHS_FR = {
        "janvier": 1, "fevrier": 2, "février": 2, "mars": 3, "avril": 4,
        "mai": 5, "juin": 6, "juillet": 7, "aout": 8, "août": 8,
        "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12, "décembre": 12,
    }
    norm = text.lower()
    now = datetime.now(timezone.utc)
    base_date = None

    if "apres-demain" in norm or "après-demain" in norm:
        base_date = (now + timedelta(days=2)).date()
    elif "demain" in norm:
        base_date = (now + timedelta(days=1)).date()

    date_match = re.search(
        r"(?:le\s+)?(?P<day>\d{1,2})\s+(?P<month>[a-zéèêûôîäëïöü]+)(?:\s+(?P<year>\d{4}))?",
        norm,
    )
    if date_match:
        day = int(date_match.group("day"))
        month = MONTHS_FR.get(date_match.group("month"))
        if not month:
            return None
        year = int(date_match.group("year")) if date_match.group("year") else now.year
        base_date = datetime(year, month, day, tzinfo=timezone.utc).date()

    if base_date is None:
        return None

    time_match = re.search(r"(?P<h>\d{1,2})h(?P<m>\d{2})?", norm)
    if not time_match:
        time_match = re.search(r"(?P<h>\d{1,2}):(?P<m>\d{2})", norm)
    hour = int(time_match.group("h")) if time_match else 0
    minute = int(time_match.group("m") or 0) if time_match else 0

    try:
        dt = datetime(base_date.year, base_date.month, base_date.day, hour, minute, tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(dt.timestamp())


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/resolve-audio")
async def resolve_audio(file: UploadFile = File(...)):
    """Receive audio, transcribe, correct station names, resolve itinerary."""
    # Save uploaded audio to a temp file
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    log.info("Received audio: %s (%d bytes, format=%s)", file.filename, len(content), suffix)

    try:
        # 1. Transcribe
        transcriber = _get_transcriber()
        transcription = transcriber.transcribe(tmp_path)
        raw_text = transcription.text
        log.info("Transcription: %r (duration=%.1fs, segments=%d)",
                 raw_text, transcription.duration_sec, len(transcription.segments))

        # 2. Phonetic correction
        corrected_text = raw_text
        corrections: list[dict] = []
        if raw_text.strip():
            corrector = _get_corrector()
            correction = corrector.correct(raw_text)
            corrected_text = correction.corrected_text
            corrections = [
                {
                    "original": c.original,
                    "corrected": c.corrected,
                    "distance": round(c.ipa_distance, 3),
                }
                for c in correction.corrections
            ]
            if corrections:
                log.info("Corrections: %s", corrections)

        # 3. Resolve itinerary
        resolver = _get_resolver()
        target_ts = _parse_datetime_from_text(corrected_text)
        order = resolver.resolve_order("audio_web", corrected_text, target_ts=target_ts)
        log.info("Resolve: valid=%s, dep=%s, arr=%s",
                 order.is_valid, order.departure_id, order.arrival_id)

        result = {
            "transcription": raw_text,
            "corrected_text": corrected_text,
            "corrections": corrections,
            "is_valid": order.is_valid,
            "departure": None,
            "arrival": None,
            "departure_time": None,
            "arrival_time": None,
            "duration_min": None,
            "path": [],
            "explored_edges": [],
        }

        if order.is_valid:
            graph = resolver.graph
            result["departure"] = resolver.id_to_name[order.departure_id]
            result["arrival"] = resolver.id_to_name[order.arrival_id]
            if order.departure_ts is not None:
                dep_dt = datetime.fromtimestamp(order.departure_ts, tz=timezone.utc)
                result["departure_time"] = dep_dt.strftime("%Y-%m-%d %H:%M")
                if order.duration_min is not None:
                    arr_dt = dep_dt + timedelta(minutes=order.duration_min)
                    result["arrival_time"] = arr_dt.strftime("%Y-%m-%d %H:%M")
                    result["duration_min"] = round(order.duration_min)

            # Full path with coordinates
            if order.path:
                result["path"] = [
                    {
                        "id": sid,
                        "name": graph.nodes[sid].get("name", sid),
                        "lat": graph.nodes[sid].get("lat"),
                        "lon": graph.nodes[sid].get("lon"),
                    }
                    for sid in order.path
                ]

            # Explored edges for Leaflet visualization (cap at 10k)
            if order.explored_edges:
                explored = []
                for from_id, to_id in order.explored_edges[:10000]:
                    fn = graph.nodes.get(from_id, {})
                    tn = graph.nodes.get(to_id, {})
                    if fn.get("lat") and tn.get("lat"):
                        explored.append({
                            "from": [fn["lat"], fn["lon"]],
                            "to": [tn["lat"], tn["lon"]],
                        })
                result["explored_edges"] = explored

        return JSONResponse(content=result)

    except Exception:
        log.exception("Error processing audio")
        raise

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
