"""
pipeline.py -- End-to-end orchestrator for object detection

Mirrors the OCR pipeline architecture:
- Single disk read per frame (reuses ocr.engine.OCREngine utilities)
- Same skip heuristics: stride, blank, blur, pHash similarity
- Outputs JSON with the same frame_id structure

Skip heuristics (cheapest first):
  1. Stride     -- fixed interval sampling
  2. Blank      -- mean luma check          (~0.1 ms)
  3. Blur       -- Laplacian variance       (~0.3 ms)
  4. Similarity -- pHash hamming distance   (~0.5 ms)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .loader import load_frames
from .utils import (
    read_frame,
    is_blank_frame,
    is_blurry,
    phash,
    hamming_distance,
)
from .engine import DetectionEngine
from .formatter import build_record, save_output

logger = logging.getLogger(__name__)

# Ghi checkpoint (đè output_file) mỗi CHECKPOINT_EVERY frame — chạy lại sau
# khi crash chỉ mất tối đa từng này frame, không mất toàn bộ tiến độ.
CHECKPOINT_EVERY = 5000


def _load_checkpoint(output_file: str) -> dict[str, dict]:
    """Đọc output_file cũ (nếu có) để biết frame nào đã xử lý xong, phục vụ resume."""
    path = Path(output_file)
    if not path.exists():
        return {}
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        return {r["frame_id"]: r for r in existing}
    except (json.JSONDecodeError, OSError):
        logger.warning("Không đọc được checkpoint cũ ở %s, bắt đầu lại từ đầu.", output_file)
        return {}


def run_pipeline(cfg: dict[str, Any]) -> None:
    """
    Run the full object detection pipeline with frame-skip optimisations.

    Parameters
    ----------
    cfg : dict
        Parsed content of ``config_detect.yaml``.
    """
    input_dir: str   = cfg["input_dir"]
    output_file: str = cfg["output_file"]
    det_cfg: dict    = cfg["detection"]
    skip_cfg: dict   = cfg.get("skip", {})

    stride: int       = max(1, int(skip_cfg.get("stride", 1)))
    blank_thresh: int  = int(skip_cfg.get("blank_brightness_threshold", 15))
    blur_thresh: float = float(skip_cfg.get("blur_threshold", 50.0))
    sim_thresh: int    = int(skip_cfg.get("similarity_hamming_threshold", 8))

    # -- 1. Discover frames ---------------------------------------------------
    frames = load_frames(input_dir)
    total = len(frames)

    # -- 1b. Resume từ checkpoint cũ (nếu có) ----------------------------------
    done = _load_checkpoint(output_file)
    if done:
        print(f"[->] Resume: {len(done)}/{total} frame đã xử lý xong ở lần chạy trước, bỏ qua.")

    # -- 2. Initialise engine -------------------------------------------------
    print(f"[->] Found {total} frames  (stride={stride})")
    print("[->] Loading YOLO model ...")
    engine = DetectionEngine(det_cfg)

    # -- 3. Process frames ----------------------------------------------------
    records: list[dict] = []
    t_start = time.perf_counter()

    skipped_stride  = 0
    skipped_blank   = 0
    skipped_blur    = 0
    skipped_similar = 0
    prev_hash = None  # pHash of the last *processed* frame
    prev_record: dict | None = None  # record of the last *processed* frame

    for idx, (frame_id, path) in enumerate(
        tqdm(frames, desc="Detect", unit="frame", dynamic_ncols=True)
    ):
        # -- Checkpoint: mỗi vòng lặp append đúng 1 record (mọi nhánh bên dưới
        # đều append+continue) nên idx == len(records) tại đây, dùng idx để tránh
        # phải sửa từng nhánh skip riêng lẻ.
        if idx > 0 and idx % CHECKPOINT_EVERY == 0:
            save_output(records, output_file)

        # -- Resume: frame này đã có kết quả từ lần chạy trước ------------------
        if frame_id in done:
            records.append(done[frame_id])
            continue

        # -- Skip: stride ------------------------------------------------------
        if idx % stride != 0:
            skipped_stride += 1
            records.append(build_record(frame_id, []))
            continue

        # -- Single disk read --------------------------------------------------
        frame_data = read_frame(path)
        if frame_data is None:
            logger.warning("Could not read frame: %s", path)
            records.append(build_record(frame_id, []))
            continue
        bgr, gray = frame_data

        # -- Skip: blank frame (luma check, ~0.1 ms) --------------------------
        if blank_thresh > 0 and is_blank_frame(gray, blank_thresh):
            logger.debug("Blank frame skipped: %s", frame_id)
            skipped_blank += 1
            records.append(build_record(frame_id, []))
            continue

        # -- Skip: blurry frame (Laplacian, ~0.3 ms) --------------------------
        if blur_thresh > 0 and is_blurry(gray, blur_thresh):
            logger.debug("Blurry frame skipped: %s", frame_id)
            skipped_blur += 1
            records.append(build_record(frame_id, []))
            continue

        # -- Skip: near-duplicate (pHash, ~0.5 ms) ----------------------------
        if sim_thresh > 0:
            curr_hash = phash(gray)
            if prev_hash is not None and curr_hash is not None:
                dist = hamming_distance(prev_hash, curr_hash)
                if dist <= sim_thresh:
                    logger.debug(
                        "Similar frame skipped (hamming=%d): %s", dist, frame_id
                    )
                    skipped_similar += 1
                    if prev_record is not None:
                        records.append({"frame_id": frame_id, "objects": prev_record["objects"]})
                    continue
            prev_hash = curr_hash

        # -- Run YOLO (GPU inference) ------------------------------------------
        detections = engine.run_with_meta(bgr)
        record = build_record(frame_id, detections)
        records.append(record)
        prev_record = record

    # -- 4. Stats & save -------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    total_skipped = skipped_stride + skipped_blank + skipped_blur + skipped_similar
    processed = total - total_skipped
    fps = processed / elapsed if elapsed > 0 else float("inf")

    print(
        f"\n[->] Frames total={total} | processed={processed} | "
        f"skipped stride={skipped_stride}, blank={skipped_blank}, "
        f"blur={skipped_blur}, similar={skipped_similar}"
    )
    print(f"[->] GPU inference: {processed} frames in {elapsed:.1f}s  ({fps:.1f} fps)")

    save_output(records, output_file)
