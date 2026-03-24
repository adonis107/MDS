#!/usr/bin/env python
"""
CLI entry point for the Market Manipulation Detection pipeline.

Usage examples
--------------
    # Run with defaults
    python run.py

    # Specify a config file
    python run.py -c config/default.yaml

    # Override model type and epochs from the command line
    python run.py --model pnn --epochs 100

    # Change data path and device
    python run.py --data data/other_book.parquet --device cpu
"""

import argparse
import logging
import sys
from typing import Dict

from detection.pipeline import AnomalyDetectionPipeline, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with timestamped output to stdout."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Build and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Market Manipulation Detection Pipeline",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML configuration file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override: path to LOB data file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["transformer_ocsvm", "pnn", "prae"],
        help="Override: model type",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default=None,
        choices=["minmax", "standard", "quantile"],
        help="Override: scaler method",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override: maximum training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override: learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override: training batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override: compute device (auto | cpu | cuda)",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        default=None,
        choices=["pot", "spot", "dspot", "rfdr"],
        help="Override: threshold method",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override: output directory for results",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config overrides
# ---------------------------------------------------------------------------

def apply_overrides(config: Dict, args: argparse.Namespace) -> Dict:
    """Merge CLI overrides into the loaded configuration dictionary."""
    if args.data is not None:
        config.setdefault("data", {})["filepath"] = args.data
    if args.model is not None:
        config.setdefault("model", {})["type"] = args.model
    if args.scaler is not None:
        config.setdefault("preprocessing", {})["scaler"] = args.scaler
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.device is not None:
        config.setdefault("training", {})["device"] = args.device
    if args.threshold is not None:
        config.setdefault("threshold", {})["method"] = args.threshold
    if args.output_dir is not None:
        config.setdefault("output", {})["results_dir"] = args.output_dir
    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments, load config, run the full pipeline."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        logger.info("Loading configuration from: %s", args.config)
        config = load_config(args.config)
        config = apply_overrides(config, args)

        pipeline = AnomalyDetectionPipeline(config)
        results = pipeline.run()

        logger.info("Pipeline completed successfully.")
        logger.info("Final results: %s", results)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.critical(
            "Pipeline failed with an unhandled exception.", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
