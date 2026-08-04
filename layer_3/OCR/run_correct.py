"""
run_correct.py -- CLI entry point, stage 2 (ising-calibration diacritic correction)

Usage
-----
    python run_correct.py                                     # reads NVIDIA_API_KEY from .env
    NVIDIA_API_KEY=... python run_correct.py                  # or from the shell env
    python run_correct.py --config my_config.yaml --api-key nvapi-...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def _load_dotenv(path: str = ".env") -> None:
    """Minimal KEY=VALUE loader -- avoids a python-dotenv dependency (pip install
    is blocked on this shared host, see memory shared-host-no-pip-install)."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ising-calibration correction stage of the OCR pipeline")
    parser.add_argument("--config", default="config.yaml", metavar="PATH")
    parser.add_argument("--frames", default=None, metavar="DIR", help="Override correct.frames_dir")
    parser.add_argument("--paddle-output", default=None, metavar="FILE", help="Override correct.paddle_output")
    parser.add_argument("--output", default=None, metavar="FILE", help="Override correct.output_file")
    parser.add_argument("--workers", type=int, default=None, help="Override correct.workers (default 6, the measured sustainable sweet spot)")
    parser.add_argument("--api-key", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[x] Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["correct"]

    if args.frames:
        cfg["frames_dir"] = args.frames
    if args.paddle_output:
        cfg["paddle_output"] = args.paddle_output
    if args.output:
        cfg["output_file"] = args.output
    if args.workers is not None:
        cfg["workers"] = args.workers

    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("[x] Set NVIDIA_API_KEY env var or pass --api-key", file=sys.stderr)
        sys.exit(1)

    from ocr.pipeline import run_correct_pipeline
    run_correct_pipeline(cfg, api_key)


if __name__ == "__main__":
    main()
