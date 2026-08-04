"""
engine.py -- YOLO inference wrapper (ultralytics)

Supports any YOLO model file: YOLOv8, YOLOv9, YOLOv10, YOLO11, or a custom
trained .pt checkpoint.  Switch models by changing ``model_path`` in config.

Example model paths:
  "yolo11n.pt"         -> YOLO11 nano  (fastest, lowest accuracy)
  "yolo11s.pt"         -> YOLO11 small
  "yolo11m.pt"         -> YOLO11 medium
  "yolo11l.pt"         -> YOLO11 large
  "yolo11x.pt"         -> YOLO11 extra-large (slowest, highest accuracy)
  "yolov8n.pt"         -> YOLOv8 nano
  "./custom_model.pt"  -> your own trained model
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DetectionEngine:
    """
    Thin wrapper around ultralytics YOLO that:
    - Loads any YOLO model from a single config value (model_path)
    - Filters results by confidence and IoU thresholds
    - Returns either plain label strings or rich dicts (label + confidence + bbox)
    - Accepts both file paths and numpy arrays as input
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """
        Parameters
        ----------
        cfg : dict
            Section ``detection`` from config_detect.yaml.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. "
                "Run: pip install ultralytics"
            ) from exc

        model_path: str = cfg.get("model_path", "yolo11n.pt")
        self.confidence_threshold: float = float(cfg.get("confidence_threshold", 0.25))
        self.iou_threshold: float = float(cfg.get("iou_threshold", 0.45))
        self.max_det: int = int(cfg.get("max_det", 300))
        self.classes: list[int] | None = cfg.get("classes", None)

        # "cuda:0" chứ không phải "0": model.to() đi thẳng vào torch.nn.Module.to()
        # (chỉ nhận device string hợp lệ), khác với predict(device=0) của ultralytics.
        # Container luôn thấy đúng 1 GPU do docker run --gpus "device=$GPU_ID" → index 0.
        device: str = "cuda:0" if cfg.get("use_gpu", True) else "cpu"
        self.imgsz: int = int(cfg.get("imgsz", 640))
        self.half: bool = cfg.get("half", True) and device != "cpu"

        logger.info(
            "DetectionEngine | model=%s | device=%s | imgsz=%d | conf=%.2f | iou=%.2f",
            model_path, device, self.imgsz,
            self.confidence_threshold, self.iou_threshold,
        )

        self._model = YOLO(model_path)
        self._model.to(device)
        # Warmup with a dummy image
        self._model.predict(
            np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8),
            imgsz=self.imgsz, verbose=False,
        )
        logger.info("YOLO engine ready (%s).", model_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, image: "str | np.ndarray") -> list[str]:
        """
        Run detection and return a plain list of label strings.

        Parameters
        ----------
        image : str | np.ndarray
            File path **or** a BGR numpy array.

        Returns
        -------
        list[str]
            Detected class names above the confidence threshold.
        """
        return [r["label"] for r in self.run_with_meta(image)]

    def run_with_meta(self, image: "str | np.ndarray") -> list[dict]:
        """
        Run detection and return full metadata per detected object.

        Parameters
        ----------
        image : str | np.ndarray
            File path **or** a BGR numpy array.

        Returns
        -------
        list[dict]
            Each element::

                {
                    "label":      str,
                    "confidence": float,   # rounded to 4 decimal places
                    "bbox":       list     # [x1, y1, x2, y2] in pixels
                }

            Empty list when nothing is detected or on error.
        """
        label = image if isinstance(image, str) else "<ndarray>"
        try:
            results = self._model.predict(
                image,
                imgsz=self.imgsz,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                classes=self.classes,
                half=self.half,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Detection failed for %s: %s", label, exc)
            return []

        records: list[dict] = []

        if not results:
            return records

        result = results[0]  # single image -> single result
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return records

        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names.get(cls_id, str(cls_id))
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            records.append({
                "label": cls_name,
                "confidence": round(conf, 4),
                "bbox": [round(v, 1) for v in xyxy],
            })

        return records
