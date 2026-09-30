"""Central logger factory — plain rotating file + console logger for app diagnostics.
This is NOT the compliance audit trail (see src/security/audit_logger.py for that)."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_PATH = Path("data/logs/vantage.log")
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


class LoggerManager:
    _logger: logging.Logger | None = None

    @classmethod
    def get_logger(cls) -> logging.Logger:
        if cls._logger is not None:
            return cls._logger

        logger = logging.getLogger("vantage")
        logger.setLevel(logging.INFO)

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

        file_handler = RotatingFileHandler(_LOG_PATH, maxBytes=5_000_000, backupCount=3)
        file_handler.setFormatter(fmt)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        cls._logger = logger
        return logger
