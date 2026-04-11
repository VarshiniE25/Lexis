"""
logger.py — Centralized logging setup using loguru.
"""

import sys
from loguru import logger as _loguru_logger
from .config import LOG_LEVEL, LOG_FILE

# Remove default handler
_loguru_logger.remove()

# Console handler
_loguru_logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
    colorize=True,
)

# File handler
_loguru_logger.add(
    str(LOG_FILE),
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
)


def get_logger(name: str):
    return _loguru_logger.bind(name=name)
