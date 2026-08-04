"""
Object Detection Module for AIC2026
Detects objects in video frames using YOLO (ultralytics).
Model is swappable via config -- any YOLO .pt file works.
"""

from .pipeline import run_pipeline
from .engine import DetectionEngine
from .formatter import build_record, save_output

__all__ = ["run_pipeline", "DetectionEngine", "build_record", "save_output"]
