# Object Detection Module Walkthrough (Layer 3, AIC2026)

Purpose: detect objects in video frames using YOLO (ultralytics) — any
YOLOv8/v9/v10/v11 model, official or custom-trained, swappable via a single
config value. Architecturally a sibling of the [OCR module](../OCR/walkthrough.md):
same skip heuristics, same frame_id scheme, same JSON-per-frame shape, YOLO
in place of PaddleOCR.

## Files

```
ObjectDetection/
├── run_detect.py            CLI entry point
├── config.yaml              all tunable settings
├── requirements.txt         ultralytics, opencv-python-headless, tqdm, pyyaml
└── object_detection/        package (name matches import — no mismatch here)
    ├── __init__.py
    ├── pipeline.py           orchestrator
    ├── loader.py             frame folder / zip scanner
    ├── engine.py             YOLO wrapper
    ├── formatter.py          JSON record builder + writer
    └── utils.py              frame I/O + skip-check helpers
```

Note: unlike the OCR module, there is no `preprocessor.py` here — frames go
straight from `read_frame` into YOLO with no CLAHE/ROI step.

## End-to-end flow (`object_detection/pipeline.py: run_pipeline`)

```
config.yaml
   │
   ▼
load_frames(input_dir)                 object_detection/loader.py
   → dir OR .zip (auto-extracted to "<zip_stem>_extracted")
   → rglob (recurses into subfolders, unlike OCR's flat iterdir)
   → sorted by filename, frame_id = file stem
   │
   ▼
DetectionEngine(det_cfg)               constructed once, before the loop
   → loads YOLO model, moves to device, runs one dummy-image warmup predict
   │
   ▼
for idx, (frame_id, path) in frames:
   │
   ├─ 1. stride check        idx % stride != 0             → skip (empty record)
   ├─ 2. utils.read_frame()  single cv2.imread → (bgr, gray), reused below
   ├─ 3. is_blank_frame(gray) mean luma < threshold         → skip
   ├─ 4. is_blurry(gray)     Laplacian variance < thresh    → skip
   ├─ 5. phash(gray) vs prev hamming distance <= thresh     → reuse prev_record's objects
   │
   ├─ engine.run_with_meta(bgr)   YOLO .predict(), GPU inference (no preprocessing step)
   └─ build_record(frame_id, detections)
   │
   ▼
save_output(records, output_file)      object_detection/formatter.py
   → JSON array, UTF-8, ensure_ascii=False, pretty-printed
```

Same "read the frame once" discipline as OCR: `read_frame` returns `(bgr, gray)`
and every skip check + the detector reuse those arrays instead of re-reading
the file.

## Module-by-module

### `object_detection/loader.py`
- `load_frames(input_path) -> list[(frame_id, abs_path)]`.
- Accepts either a directory **or a `.zip` file** — a capability the OCR
  loader doesn't have. A zip is extracted once to `<zip_stem>_extracted`
  next to it (skipped on subsequent runs if that folder already exists), then
  treated as the input directory.
- Uses `rglob("*")` (recursive), not `iterdir()` — so nested subfolders (e.g.
  a zip that unpacks into a subdirectory) are picked up, unlike OCR's
  `loader.py` which only scans the top level.
- Same supported extensions as OCR (`.jpg .jpeg .png .bmp .webp .tiff .tif`),
  same `FileNotFoundError` / `NotADirectoryError` / `ValueError` raised on bad
  input and left to propagate (no try/except in `run_detect.py` either).

### `object_detection/utils.py`
- Standalone reimplementation of the OCR module's frame-I/O and skip-check
  statics (`read_frame`, `is_blank_frame`, `is_blurry`, `phash`,
  `hamming_distance`) — byte-for-byte the same logic as
  `ocr/engine.py`'s static methods, just as free functions instead of
  `OCREngine` staticmethods.
- Note: `pipeline.py`'s module docstring (line 5) says it "reuses
  `ocr.engine.OCREngine` utilities" — that's stale/inaccurate; the code
  actually imports from this local `utils.py`, not from the OCR package.
  There's no cross-package dependency between `ObjectDetection` and `OCR`.

### `object_detection/engine.py` — `DetectionEngine`
- Wraps `ultralytics.YOLO`, constructed once from the `detection:` config
  section.
- `model_path` is the only thing you change to swap models — official names
  (`yolo11n.pt` ... `yolo11x.pt`, `yolov8n.pt`, ...) are auto-downloaded by
  ultralytics on first use; a local path (`"./my_model.pt"`) loads a custom
  checkpoint.
- `device = "0" if use_gpu else "cpu"` — hardcoded to GPU index `0`; there's
  no config knob for selecting a different GPU (unlike OCR, which delegates
  device selection entirely to PaddleOCR's own `use_gpu` flag).
- `half` (FP16) is force-disabled when `device == "cpu"`, since half precision
  isn't meaningful on CPU.
- Constructor does a one-time warmup: `self._model.predict(np.zeros((imgsz,
  imgsz, 3), ...))` — pays the first-call JIT/graph-compile cost once, up
  front, rather than on the first real frame.
- `run_with_meta(image)` calls `.predict(...)` with `imgsz`, `conf`, `iou`,
  `max_det`, `classes`, `half`; iterates `result.boxes`, returns
  `[{"label", "confidence", "bbox": [x1,y1,x2,y2]}, ...]`. `bbox` here is
  **always included** (no `include_bbox` toggle like OCR has).
- Any exception during `.predict()` is caught and logged, returning `[]`
  rather than aborting the whole run — same defensive pattern as
  `OCREngine.run_with_meta`.
- `run(image)` is a thin convenience wrapper returning just label strings;
  unused by `pipeline.py` (same unused-helper pattern as OCR's `OCREngine.run`).

### `object_detection/formatter.py`
- `build_record(frame_id, detections)` → `{"frame_id", "objects": detections}`.
  `detections` (from `DetectionEngine.run_with_meta`) already has the exact
  `{"label", "confidence", "bbox"}` shape, so it's passed through directly —
  no `include_bbox` flag, no per-object copy loop (removed; it used to
  rebuild each dict field-by-field for no reason).
- `save_output` — identical shape to OCR's: creates parent dirs, UTF-8,
  `ensure_ascii=False`, 2-space indent.

### `object_detection/pipeline.py`
- Ties loader → skip checks → engine → formatter together; tracks and prints
  the same `stride/blank/blur/similar` skip counters and FPS as OCR's pipeline.
- Structurally near-identical to `ocr/pipeline.py`, minus the preprocessing
  step (there's nothing between `read_frame` and `engine.run_with_meta`).
- `prev_record` is tracked alongside `prev_hash`, set at the same point (right
  after a frame is genuinely run through YOLO). The near-duplicate branch
  reuses `prev_record["objects"]`, **not** `records[-1]` — `records[-1]` can be
  an unrelated blank/blur placeholder appended between the two hash-compared
  frames, so reading from `prev_record` guarantees the reused detections
  always come from the actual frame `prev_hash` was computed from. (Previously
  this read `records[-1]` directly, which could silently copy an empty result
  onto a frame that was really a duplicate of an earlier, real detection.)
- Stride-skipped frames now append an empty-`objects` placeholder record too
  (previously they were omitted from the output entirely, unlike blank/blur/
  similar skips) — every `frame_id` in `input_dir` now has exactly one entry
  in the output JSON, matching the "one entry per frame" contract below.

### `run_detect.py`
- Argparse CLI: `--config`, `--input`, `--output`, `--model`, `--no-gpu`,
  `--verbose`.
- `--model` overrides `detection.model_path` directly — the fastest way to
  swap YOLO variants without editing the YAML.
- `--no-gpu` sets `use_gpu=False` and `half=False`.
- Imports `object_detection.pipeline` (matches the actual package name — no
  import/directory mismatch here).
- `pipeline` import is deferred inside `main()`, same rationale as OCR: keep
  `--help` fast without importing ultralytics/torch just to print usage.

## Config surface (`config.yaml`)

| Section | Key | Effect |
|---|---|---|
| top-level | `input_dir`, `output_file` | I/O paths (no `include_bbox` — bbox always emitted) |
| `skip` | `stride` | process every Nth frame |
| `skip` | `blank_brightness_threshold` | 0 disables the blank check |
| `skip` | `blur_threshold` | 0 disables the blur check |
| `skip` | `similarity_hamming_threshold` | 0 disables pHash dedup |
| `detection` | `model_path` | YOLO checkpoint — official name (auto-download) or local path |
| `detection` | `use_gpu`, `half` | device selection ("0" or "cpu") + FP16 |
| `detection` | `imgsz` | square inference resolution (default 640) |
| `detection` | `confidence_threshold`, `iou_threshold` | detection score cutoff + NMS overlap threshold |
| `detection` | `max_det` | cap on detections per frame |
| `detection` | `classes` | `null` = all COCO classes, or a list of class IDs to keep (e.g. `[0]` = person only) |

## How to run

```bash
pip install -r requirements.txt   # ultralytics auto-installs a matching torch build
python run_detect.py                                    # config.yaml -> ./detections.json
python run_detect.py --input ./frames.zip --output ./out.json
python run_detect.py --model yolo11m.pt                  # swap model for one run
python run_detect.py --no-gpu --verbose                  # CPU debug run
```

Output is a JSON array of `{"frame_id", "objects": [{"label", "confidence",
"bbox": [x1,y1,x2,y2]}]}`, one entry per frame, in sorted-filename order
(recursive if the input was a directory tree or zip with nested folders).
