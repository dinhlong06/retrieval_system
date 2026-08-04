"""
formatter.py -- Output JSON schema builder for object detection

Output schema (per frame)
-------------------------
{
    "frame_id": "L21_VID01_0001",
    "objects": [
        {"label": "person", "confidence": 0.9512, "bbox": [120.5, 340.2, 450.8, 720.0]},
        ...
    ]
}
"""

from __future__ import annotations

import json
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_record(
    frame_id: str,
    detections: list[dict],
) -> dict:
    """
    Build a single output record.

    Parameters
    ----------
    frame_id : str
        Unique identifier for the frame (file stem).
    detections : list[dict]
        Raw output from ``DetectionEngine.run_with_meta()``.
        Each element: ``{"label": str, "confidence": float, "bbox": list}``.

    Returns
    -------
    dict
        ``{"frame_id": str, "objects": list[dict]}``
    """
    return {
        "frame_id": frame_id,
        "objects": detections,
    }


def save_output(records: list[dict], output_path: str) -> None:
    """
    Serialise *records* to a JSON file at *output_path*.

    Parameters
    ----------
    records : list[dict]
        List of frame records from :func:`build_record`.
    output_path : str
        Destination file path (created or overwritten).
    """
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Ghi .tmp rồi replace: checkpoint bị kill giữa chừng sẽ để lại JSON cụt,
    # _load_checkpoint bắt JSONDecodeError và trả {} = mất sạch tiến độ trong im lặng.
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(out)

    logger.info("Saved %d records -> %s", len(records), out)
