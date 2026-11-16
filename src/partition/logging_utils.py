"""Journalisation : console pour l'application, fichier ``job.log`` par job."""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
ROOT_LOGGER = "partition"


def setup_logging(level: int | str = logging.INFO) -> logging.Logger:
    """Configure la journalisation console de l'application (idempotent)."""
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(level)
    logger.propagate = False  # évite les doublons si une bibliothèque configure le logger racine
    if not any(getattr(h, "_partition_console", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handler._partition_console = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


def get_job_logger(job_dir: Path, job_id: str | None = None) -> logging.Logger:
    """Retourne un logger qui écrit dans ``<job_dir>/job.log`` (et remonte vers la console)."""
    job_dir.mkdir(parents=True, exist_ok=True)
    name = f"{ROOT_LOGGER}.job.{job_id or job_dir.name}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_file = (job_dir / "job.log").resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_file:
            return logger
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    return logger


def close_job_logger(logger: logging.Logger) -> None:
    """Ferme les fichiers ouverts par :func:`get_job_logger` (indispensable sous Windows)."""
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)
