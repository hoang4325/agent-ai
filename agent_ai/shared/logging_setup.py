"""Shared logging configuration helpers."""
from __future__ import annotations

import logging


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(
    *,
    level: int = logging.INFO,
    format: str = DEFAULT_LOG_FORMAT,
    force: bool = False,
) -> None:
    """
    Configure root logging once with a consistent format.

    ``force=True`` reconfigures even if handlers already exist (Python 3.8+).
    """
    kwargs: dict = {
        "level": level,
        "format": format,
    }
    # ``force`` is available on Python 3.8+.
    try:
        logging.basicConfig(**kwargs, force=force)
    except TypeError:
        logging.basicConfig(**kwargs)
