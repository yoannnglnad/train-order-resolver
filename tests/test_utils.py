from pathlib import Path

from src.utils.cache import Cache
from src.utils.logging import get_json_logger, log_decision


def test_cache_round_trip(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.sqlite"
    with Cache(cache_path) as cache:
        cache.set("hello", "world")
        assert cache.get("hello") == "world"
        cache.clear()
        assert cache.get("hello") is None


def test_json_logger_writes_file(tmp_path: Path) -> None:
    log_path = tmp_path / "logs.jsonl"
    logger = get_json_logger(name="tor-test", log_path=log_path, stream_to_stdout=False)
    log_decision(logger, sentence_id="1", decision={"depart": "A", "arrivee": "B"}, score=0.9, latency_ms=12.0)
    logger.handlers.clear()

    content = log_path.read_text(encoding="utf-8")
    assert '"sentence_id": "1"' in content
    assert '"decision"' in content
