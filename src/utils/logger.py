"""Centralised logging for the whole application.

``get_logger(__name__)`` is the only supported way to obtain a logger: it keeps
the format, level and handlers consistent between the Streamlit app, the CLI and
the test-suite, and it never double-configures the root logger.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
_CONFIGURED: bool = False


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None, *, force: bool = False) -> None:
    """Configure the ``pushup`` root logger exactly once.

    Args:
        level: Logging level name (``"DEBUG"``, ``"INFO"``, ...) or level int.
        log_file: Optional file to mirror the log output into.
        force: Re-configure even if logging was already set up.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    logger = logging.getLogger("pushup")
    logger.setLevel(level if isinstance(level, int) else level.upper())
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        try:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            logger.warning("Could not attach log file %s: %s", log_file, exc)

    _CONFIGURED = True


def get_logger(name: str, level: Optional[str | int] = None) -> logging.Logger:
    """Return a namespaced child of the ``pushup`` logger.

    Args:
        name: Usually ``__name__`` of the calling module.
        level: Optional level override for this logger only.
    """
    if not _CONFIGURED:
        setup_logging("INFO")
    short = name.split(".")[-1] if name.startswith("src.") else name
    logger = logging.getLogger(f"pushup.{short}")
    if level is not None:
        logger.setLevel(level if isinstance(level, int) else level.upper())
    return logger
