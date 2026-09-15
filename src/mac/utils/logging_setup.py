"""System logging setup."""
from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{experiment_name}.log"
    logger = logging.getLogger("fusion")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
