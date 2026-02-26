"""Build and load a phonetic (IPA) index for all SNCF stations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import polars as pl
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

from src.utils.config import DEFAULT_DATASET_PATH, DEFAULT_PHONETIC_INDEX_PATH


def _get_backend() -> EspeakBackend:
    return EspeakBackend(
        language="fr-fr",
        preserve_punctuation=False,
        with_stress=False,
        language_switch="remove-flags",
    )


def _phonemize_batch(texts: List[str], backend: EspeakBackend) -> List[str]:
    """Phonemize a list of strings, returning IPA for each."""
    separator = Separator(phone=" ", word=" _ ", syllable="")
    results = backend.phonemize(texts, separator=separator, strip=True)
    # Normalize whitespace left by language-switch flag removal
    return [" ".join(r.split()) for r in results]


def build_phonetic_index(
    stations_path: Path = DEFAULT_DATASET_PATH,
    output_path: Path = DEFAULT_PHONETIC_INDEX_PATH,
) -> Dict[str, dict]:
    """Build IPA index from stations.parquet and save as JSON.

    Each entry maps station_id to:
      - name: original station name
      - name_norm: normalized name
      - city: city name
      - ipa_name: IPA transcription of name
      - ipa_name_norm: IPA transcription of name_norm
      - ipa_city: IPA transcription of city
    """
    df = pl.read_parquet(stations_path)
    # Exclude interpolated haltes (not real passenger stations)
    df = df.filter(pl.col("passengers") != "I")
    backend = _get_backend()

    station_ids = df["station_id"].to_list()
    names = df["name"].to_list()
    names_norm = df["name_norm"].to_list()
    cities = df["city"].to_list()

    # Phonemize all fields in batch for speed
    all_texts = names + names_norm + cities
    all_ipa = _phonemize_batch(all_texts, backend)

    n = len(station_ids)
    ipa_names = all_ipa[:n]
    ipa_names_norm = all_ipa[n : 2 * n]
    ipa_cities = all_ipa[2 * n :]

    index: Dict[str, dict] = {}
    for i, sid in enumerate(station_ids):
        index[sid] = {
            "name": names[i],
            "name_norm": names_norm[i],
            "city": cities[i].title() if isinstance(cities[i], str) else cities[i],
            "ipa_name": ipa_names[i],
            "ipa_name_norm": ipa_names_norm[i],
            "ipa_city": ipa_cities[i],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=0)

    return index


def load_phonetic_index(
    path: Path = DEFAULT_PHONETIC_INDEX_PATH,
) -> Dict[str, dict]:
    """Load the pre-built phonetic index from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
