"""JSON logging helpers for decision and metrics outputs."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.utils.config import DEFAULT_LOG_PATH


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Merge extra fields if present
        for key, value in record.__dict__.items():
            if key in {"msg", "args", "levelname", "levelno", "pathname", "filename", "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process", "name"}:
                continue
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def get_json_logger(
    name: str = "train_order_resolver",
    log_path: Path | None = None,
    stream_to_stdout: bool = True,
) -> logging.Logger:
    """Create a logger that emits JSON lines to file and/or stdout."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # reuse existing configuration

    logger.setLevel(logging.INFO)
    formatter = JsonFormatter()

    if stream_to_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    path = log_path or DEFAULT_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def log_decision(
    logger: logging.Logger,
    sentence_id: str,
    decision: Any,
    score: float | None,
    latency_ms: float,
) -> None:
    """Log a per-sentence decision as JSON line."""
    logger.info(
        "decision",
        extra={
            "sentence_id": sentence_id,
            "decision": decision,
            "score": score,
            "latency_ms": latency_ms,
        },
    )


def log_metrics(logger: logging.Logger, metrics: Dict[str, Any]) -> None:
    """Log aggregated metrics at end of batch."""
    logger.info("metrics", extra=metrics)
