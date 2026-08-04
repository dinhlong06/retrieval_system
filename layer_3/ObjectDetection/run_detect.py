"""
run_detect.py -- CLI entry point for object detection

Usage
-----
    python run_detect.py                              # uses default config_detect.yaml
    python run_detect.py --config my_config.yaml      # custom config file
    python run_detect.py --input ./frames --output ./detections.json
    python run_detect.py --model yolo11m.pt           # override model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Object detection on video frames using YOLO -- AIC2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--input",
        default=None,
        metavar="DIR",
        help="Override input_dir from config",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Override output_file from config",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="PATH",
        help="Override model_path (e.g. yolo11n.pt, yolo11m.pt, ./custom.pt)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU inference (overrides config)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def _load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        print(f"[x] Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    args = _parse_args()

    # -- Logging ---------------------------------------------------------------
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # -- Load config -----------------------------------------------------------
    cfg = _load_config(args.config)

    # -- Apply CLI overrides ---------------------------------------------------
    if args.input:
        cfg["input_dir"] = args.input
    if args.output:
        cfg["output_file"] = args.output
    if args.model:
        cfg.setdefault("detection", {})["model_path"] = args.model
    if args.no_gpu:
        cfg.setdefault("detection", {})["use_gpu"] = False
        cfg["detection"]["half"] = False

    # -- Run pipeline ----------------------------------------------------------
    from object_detection.pipeline import run_pipeline  # noqa: PLC0415
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
