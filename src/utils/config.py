"""Central place for CLI defaults and shared constants."""

from __future__ import annotations

from pathlib import Path

DEFAULT_INPUT_PATH = Path("inputs.txt")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_RAW_DATA_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_MODEL_DIR = DEFAULT_DATA_DIR / "models"
DEFAULT_DATASET_PATH = DEFAULT_DATA_DIR / "dataset" / "stations.parquet"
DEFAULT_CONNECTIONS_PATH = DEFAULT_DATA_DIR / "dataset" / "connections.parquet"
DEFAULT_SCHEDULE_PATH = DEFAULT_DATA_DIR / "dataset" / "full_schedule.parquet"
DEFAULT_CONNECTION_RULES_PATH = DEFAULT_DATA_DIR / "dataset" / "connection_rules.csv"
DEFAULT_PHONETIC_INDEX_PATH = DEFAULT_DATA_DIR / "cache" / "phonetic_index.json"
DEFAULT_STT_MODEL_ID = "large-v3"
DEFAULT_CACHE_PATH = DEFAULT_DATA_DIR / "cache.sqlite"
DEFAULT_LOG_PATH = DEFAULT_DATA_DIR / "logs" / "decisions.jsonl"
DEFAULT_K_NEIGHBORS = 8
