#!/usr/bin/env bash
# Build + chạy nhánh ReCap của keyframe_pipeline trong Docker.
#
# Cách dùng:
#   ./run_recap.sh                                 # caption mọi video trong $PIPELINE
#   ./run_recap.sh --overwrite                     # ghi đè .jsonl đã có
#   LIMIT_VIDEO=K01_V001 ./run_recap.sh            # chỉ 1 video (recap-video)
#   MAX_CONCURRENCY=4 ./run_recap.sh               # nhiều request song song
#   MODEL=qwen3-vl:4b-instruct ./run_recap.sh      # đổi model
#   NO_METADATA=1 ./run_recap.sh                   # bỏ hint Layer 3, caption từ ảnh trần
#   RESUME=1 ./run_recap.sh                        # chạy tiếp phần còn thiếu
#   SKIP_BUILD=1 ./run_recap.sh                    # dùng image có sẵn
#
# Input : layer_2/Keyframe_Extracting/benchmark/<PIPELINE>/<VIDEO_ID>/<keyframe_id>.jpg
#         + layer_3 OCR/ObjectDetection JSON (join theo frame_id == keyframe_id)
# Output: ./artifacts/recap/<VIDEO_ID>.jsonl, mỗi dòng đúng 4 field
#         {video_id, keyframe_id, caption, corrected_ocr}
#
# Ollama chạy NGOÀI container; script này gọi ./run_ollama.sh start để bật/kiểm
# server trước (bind 0.0.0.0, model ghim VRAM, vision, xem run_ollama.sh để biết lý do).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------- tham số chạy
IMAGE_NAME="${IMAGE_NAME:-ai26-recap}"
PIPELINE="${PIPELINE:-pipeline_c}"
FRAMES_DIR="${FRAMES_DIR:-$PROJECT_ROOT/layer_2/Keyframe_Extracting/benchmark/$PIPELINE}"
OCR_JSON="${OCR_JSON:-$PROJECT_ROOT/layer_3/OCR/output/output.json}"
OBJECTS_JSON="${OBJECTS_JSON:-$PROJECT_ROOT/layer_3/ObjectDetection/output/detections.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/artifacts/recap}"
LIMIT_VIDEO="${LIMIT_VIDEO:-}"
NO_METADATA="${NO_METADATA:-}"
RESUME="${RESUME:-}"
SKIP_BUILD="${SKIP_BUILD:-}"

# ----------------------------------------------------- siêu tham số inference
# Đổi được từ đây; đây là những thứ ảnh hưởng trực tiếp chất lượng/tốc độ caption.
MODEL="${MODEL:-qwen3-vl:8b-instruct-gpu}"   # bản num_gpu 999: từ chối nạp nếu không đủ VRAM
OLLAMA_PORT="${OLLAMA_PORT:-11501}"
OLLAMA_HOST_LOCAL="${OLLAMA_HOST_LOCAL:-http://127.0.0.1:$OLLAMA_PORT}"   # preflight từ host
OLLAMA_HOST_INSIDE="${OLLAMA_HOST_INSIDE:-http://10.0.2.3:$OLLAMA_PORT}"  # container -> host
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"

# Các hằng còn lại nằm trong code vì chúng thuộc contract của output, không phải
# nút vặn vận hành:
#   temperature=0, NUM_PREDICT=768, think=False, format=CAPTION_RESPONSE_SCHEMA
#     -> keyframe_pipeline/backends/recap_ollama.py
#   DEFAULT_RECAP_PROMPT (caption 60-100 từ + corrected_ocr)
#     -> keyframe_pipeline/config.py

mkdir -p "$OUTPUT_DIR"

# ------------------------------------------------------------------- preflight
echo "== Preflight Ollama =="
BASE_MODEL="${BASE_MODEL:-qwen3-vl:8b-instruct}" MODEL="$MODEL" OLLAMA_PORT="$OLLAMA_PORT" \
    "$SCRIPT_DIR/run_ollama.sh" start

# ----------------------------------------------------------------------- build
if [ -z "$SKIP_BUILD" ]; then
    echo "== Build image $IMAGE_NAME =="
    docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
fi

docker run --rm --entrypoint python3 "$IMAGE_NAME" -c \
    "import json,urllib.request;urllib.request.urlopen('$OLLAMA_HOST_INSIDE/api/tags',timeout=8)" \
    || { echo "LỖI: container không gọi được $OLLAMA_HOST_INSIDE." >&2
         echo "     Ollama phải bind 0.0.0.0 (xem header script)." >&2; exit 1; }

# ------------------------------------------------------------------------- run
ARGS=(--ollama-model "$MODEL" --ollama-host "$OLLAMA_HOST_INSIDE" --max-concurrency "$MAX_CONCURRENCY")
if [ -n "$RESUME" ]; then ARGS+=(--resume); fi
MOUNTS=(-v "$FRAMES_DIR:/data/frames:ro" -v "$OUTPUT_DIR:/data/output")
if [ -z "$NO_METADATA" ]; then
    MOUNTS+=(-v "$OCR_JSON:/data/ocr.json:ro" -v "$OBJECTS_JSON:/data/objects.json:ro")
    ARGS+=(--ocr-json /data/ocr.json --objects-json /data/objects.json)
fi

if [ -n "$LIMIT_VIDEO" ]; then
    echo "== ReCap video $LIMIT_VIDEO ($MODEL) =="
    docker run --rm "${MOUNTS[@]}" "$IMAGE_NAME" recap-video \
        --keyframe-dir "/data/frames/$LIMIT_VIDEO" \
        --output "/data/output/$LIMIT_VIDEO.jsonl" \
        "${ARGS[@]}" "$@"
else
    echo "== ReCap dataset $PIPELINE ($MODEL) =="
    docker run --rm "${MOUNTS[@]}" "$IMAGE_NAME" recap-dataset \
        --dataset-root /data/frames \
        --output-dir /data/output \
        "${ARGS[@]}" "$@"
fi

echo "== Xong -> $OUTPUT_DIR =="
