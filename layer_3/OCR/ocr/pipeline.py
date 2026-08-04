"""
pipeline.py -- end-to-end orchestrators for the two OCR pipeline stages

Stage 1 (run_paddle_pipeline): GPU detection+recognition (PaddleOCR PP-OCRv6).
Stage 2 (run_correct_pipeline): rate-limited API call per frame to restore
diacritics (ising-calibration VLM). Kept as two stages, not one pipeline,
because they run in different environments -- stage 1 needs the GPU/Paddle
Docker image, stage 2 only needs `requests` and a network path to NVIDIA's API.

Both stages share the same skip-heuristic idea (ported from an earlier local
Paddle+VietOCR pipeline, retired): a frame identical/near-identical to the one before it reuses that result
instead of re-running (GPU) inference or paying for another (rate-limited)
API call.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from PIL import Image

from . import frame_skip
from .corrector import correct
from .formatter import load_checkpoint, save_output
from .loader import load_frames
from .paddle_engine import PaddleEngine

# Calibrated on 60 boxed frames: a truncated ticker line's box sits at EXACTLY
# x=0 or x=frame_width (median 0), not "near" it -- a tight threshold avoids
# flagging text that's merely positioned close to the margin.
EDGE_PX = 10


def _truncated_mask(texts: list[dict], image_path: str) -> list[bool] | None:
    """None (not a list of False) when boxes are absent, so correct() can tell
    "known untruncated" apart from "unknown" instead of silently disabling
    itself on older paddle output that predates the box field."""
    if not texts or "box" not in texts[0]:
        return None
    width = Image.open(image_path).size[0]
    return [t["box"][0] <= EDGE_PX or t["box"][2] >= width - EDGE_PX for t in texts]

# The isolated 60-frame benchmark (see memory ising-calibration-api-rate-limit)
# measured 6 as the sweet spot (2.62 req/s vs 1.98 at 8), but a real 714-frame
# overnight run with this key already warmed up from a day of testing showed
# visibly bursty stalls at 6 -- lowered to 4 based on that live observation.
# Re-measure before raising back to 6+ rather than assuming the clean-state
# benchmark still holds on a heavily-used key.
DEFAULT_CORRECT_WORKERS = 4


def run_paddle_pipeline(cfg: dict) -> None:
    input_dir: str = cfg["input_dir"]
    output_file: str = cfg["output_file"]
    output_file_paddle_origin: str = cfg["output_file_paddle_origin"]
    limit: int = cfg.get("limit", 0) or None
    checkpoint_every: int = cfg.get("checkpoint_every", 0)
    skip_cfg: dict = cfg.get("skip", {})
    preprocess: bool = cfg.get("preprocess", False)

    blank_thresh = skip_cfg.get("blank_brightness_threshold", 0)
    blur_thresh = skip_cfg.get("blur_threshold", 0.0)
    sim_thresh = skip_cfg.get("similarity_hamming_threshold", 0)

    frames = load_frames(input_dir)[:limit] if limit else load_frames(input_dir)
    done = load_checkpoint(output_file) if checkpoint_every else {}
    done_origin = load_checkpoint(output_file_paddle_origin) if checkpoint_every else {}
    if done:
        print(f"[->] Resume: {len(done)}/{len(frames)} frames already done, skipping.")

    engine_cfg = cfg.get("ocr", {})
    print(f"[->] {len(frames)} frames, lang={engine_cfg.get('lang', 'vi')}, preprocess={preprocess}")
    engine = PaddleEngine(
        lang=engine_cfg.get("lang", "vi"),
        confidence_threshold=engine_cfg.get("confidence_threshold", 0.5),
        ascii_only=engine_cfg.get("ascii", False),
        ocr_version=engine_cfg.get("ocr_version"),
        unclip_ratio=engine_cfg.get("unclip_ratio"),
    )

    records: list[dict] = []
    records_origin: list[dict] = []
    skipped_blank = skipped_blur = skipped_similar = 0
    prev_hash, prev_record, prev_record_origin = None, None, None

    for idx, (frame_id, path) in enumerate(tqdm(frames, desc="paddle", unit="frame")):
        if checkpoint_every and idx > 0 and idx % checkpoint_every == 0:
            save_output(records, output_file)
            save_output(records_origin, output_file_paddle_origin)

        if frame_id in done:
            records.append(done[frame_id])
            records_origin.append(done_origin.get(frame_id, {"frame_id": frame_id, "texts": []}))
            continue

        need_gray = blank_thresh or blur_thresh or sim_thresh or preprocess
        bgr = gray = None
        if need_gray:
            frame_data = frame_skip.read_frame(path)
            if frame_data is None:
                records.append({"frame_id": frame_id, "texts": []})
                records_origin.append({"frame_id": frame_id, "texts": []})
                continue
            bgr, gray = frame_data

        if blank_thresh and frame_skip.is_blank_frame(gray, blank_thresh):
            skipped_blank += 1
            records.append({"frame_id": frame_id, "texts": []})
            records_origin.append({"frame_id": frame_id, "texts": []})
            continue
        if blur_thresh and frame_skip.is_blurry(gray, blur_thresh):
            skipped_blur += 1
            records.append({"frame_id": frame_id, "texts": []})
            records_origin.append({"frame_id": frame_id, "texts": []})
            continue
        if sim_thresh:
            curr_hash = frame_skip.phash(gray)
            if prev_hash is not None and frame_skip.hamming_distance(prev_hash, curr_hash) <= sim_thresh:
                skipped_similar += 1
                records.append({"frame_id": frame_id, "texts": prev_record["texts"] if prev_record else []})
                records_origin.append({"frame_id": frame_id, "texts": prev_record_origin["texts"] if prev_record_origin else []})
                continue
            prev_hash = curr_hash

        source = frame_skip.preprocess(bgr) if preprocess else path
        texts, texts_origin = engine.run(source)
        record = {"frame_id": frame_id, "texts": texts}
        record_origin = {"frame_id": frame_id, "texts": texts_origin}
        records.append(record)
        records_origin.append(record_origin)
        prev_record, prev_record_origin = record, record_origin

    print(f"[->] Skipped -- blank={skipped_blank}, blur={skipped_blur}, similar={skipped_similar}")
    save_output(records, output_file)
    save_output(records_origin, output_file_paddle_origin)
    print(f"[->] Saved {len(records)} records -> {Path(output_file).resolve()}")
    print(f"[->] Saved {len(records_origin)} records -> {Path(output_file_paddle_origin).resolve()}")


def run_correct_pipeline(cfg: dict, api_key: str) -> None:
    import json

    frames_dir: str = cfg["frames_dir"]
    paddle_output: str = cfg["paddle_output"]
    output_file: str = cfg["output_file"]
    checkpoint_every: int = cfg.get("checkpoint_every", 0)
    workers: int = cfg.get("workers", DEFAULT_CORRECT_WORKERS)

    paths = dict(load_frames(frames_dir))
    paddle = json.loads(Path(paddle_output).read_text(encoding="utf-8"))
    done = load_checkpoint(output_file) if checkpoint_every else {}
    if done:
        print(f"[->] Resume: {len(done)}/{len(paddle)} frames already done, skipping.")

    # Group consecutive frames with byte-identical raw text (a banner sitting
    # unchanged for seconds) so only the FIRST frame of each run pays for an
    # API call; the rest of the run reuses that result. Groups are then
    # distributed across worker threads -- order of completion doesn't matter
    # since results are written back by frame_id, not by arrival order.
    pending = [rec for rec in paddle if rec["frame_id"] not in done]
    groups: list[list[dict]] = []
    prev_raw = None
    for rec in pending:
        raw = [t["text"] for t in rec["texts"]]
        if raw and raw == prev_raw:
            groups[-1].append(rec)
        else:
            groups.append([rec])
        prev_raw = raw
    skipped_similar = sum(len(g) - 1 for g in groups)

    records_by_id: dict[str, dict] = dict(done)

    def process_group(group: list[dict]) -> tuple[list[dict], str]:
        rec = group[0]
        raw = [t["text"] for t in rec["texts"]]
        truncated = _truncated_mask(rec["texts"], paths[rec["frame_id"]])
        fixed, reason = correct(api_key, paths[rec["frame_id"]], raw, truncated)
        # Replace only the text; confidence and box ride through untouched.
        texts = [{**t, "text": new} for t, new in zip(rec["texts"], fixed)]
        return [{"frame_id": r["frame_id"], "texts": texts} for r in group], reason

    completed = 0
    last_checkpoint_at = 0
    reasons: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_group, g): g for g in groups}
        with tqdm(total=len(pending), desc="correct", unit="frame") as pbar:
            for future in as_completed(futures):
                group_records, reason = future.result()
                reasons[reason] += 1
                for record in group_records:
                    records_by_id[record["frame_id"]] = record
                pbar.update(len(futures[future]))
                completed += len(futures[future])
                if checkpoint_every and completed - last_checkpoint_at >= checkpoint_every:
                    last_checkpoint_at = completed
                    save_output([records_by_id[r["frame_id"]] for r in paddle if r["frame_id"] in records_by_id], output_file)

    print(f"[->] Skipped (identical raw text to previous frame) -- {skipped_similar}")
    print(f"[->] API calls by outcome -- {dict(reasons)}")
    records = [records_by_id[rec["frame_id"]] for rec in paddle]
    save_output(records, output_file)
    print(f"[->] Saved {len(records)} records -> {Path(output_file).resolve()}")
