"""CLI for running fusion experiments."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

import torch

from src.utils.config import load_config, merge_configs, config_hash
from src.utils.logging_setup import setup_logging
from src.utils.reporting import generate_report
from src.utils.registry import ResultsRegistry

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--fusion_config", required=True, help="Path to fusion method config"
    )
    parser.add_argument("--name", default=None, help="Experiment name")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    fusion_cfg = load_config(args.fusion_config)
    cfg = merge_configs(base_cfg, fusion_cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"{fusion_cfg.get('fusion_type', 'experiment')}_{timestamp}"

    logger = setup_logging(cfg["logging"]["log_dir"], name)
    logger.info(f"Starting experiment: {name}")
    logger.info(f"Config hash: {config_hash(cfg)}")

    # TODO: Build fusion model, dataset, and run LOSO
    # This will be completed when fusion models are implemented in Phase 3
    logger.info("Experiment runner ready. Fusion models to be added in Phase 3.")


if __name__ == "__main__":
    main()
