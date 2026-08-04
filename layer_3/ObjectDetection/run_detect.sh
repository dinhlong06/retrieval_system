#!/usr/bin/env bash
# Build + chạy Layer 3 - Object Detection (YOLO/ultralytics) trong Docker.
#
# Cách dùng:
#   ./run_detect.sh                            # chạy full mọi video trong $PIPELINE
#   ./run_detect.sh --model yolo11m.pt         # đổi model
#   ./run_detect.sh --no-gpu                   # ép chạy CPU
#   PIPELINE=pipeline_a ./run_detect.sh        # đổi pipeline nguồn
#
# Input: layer_2/Keyframe_Extracting/benchmark/<PIPELINE>/ — loader.py quét đệ quy
# nên tự gộp mọi video trong đó (tên file keyframe_id đã unique toàn cục, không lo trùng).
# Override: FRAMES_DIR=/path/khac ./run_detect.sh   (bỏ qua PIPELINE)
# Output: detections.json ghi vào ./output/ (gộp chung mọi video).
#
# Host này là GPU server dùng chung -> mặc định chọn 1 GPU đang rảnh nhất
# (free memory cao nhất) qua nvidia-smi, có thể override bằng biến GPU_ID.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE_NAME="ai26-layer3-detect"
PIPELINE="${PIPELINE:-pipeline_c}"
FRAMES_DIR="${FRAMES_DIR:-$PROJECT_ROOT/layer_2/Keyframe_Extracting/benchmark/$PIPELINE}"
OUTPUT_DIR="$SCRIPT_DIR/output"
# Không mount cache thì ultralytics tải lại weight mỗi lần chạy vào lớp ghi của
# container, tức vào ổ / của host vốn đã đầy.
mkdir -p "$OUTPUT_DIR" "$SCRIPT_DIR/cache/ultralytics" "$SCRIPT_DIR/cache/weights"

GPU_ID="${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')}"

echo "== Build image $IMAGE_NAME =="
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

echo "== Chạy Object Detection trên GPU $GPU_ID =="
docker run --rm \
    --gpus "device=$GPU_ID" \
    -v "$FRAMES_DIR:/data/frames:ro" \
    -v "$OUTPUT_DIR:/data/output" \
    -v "$SCRIPT_DIR/cache/ultralytics:/root/.config/Ultralytics" \
    -v "$SCRIPT_DIR/cache/weights:/workspace/weights" \
    "$IMAGE_NAME" \
    --input /data/frames \
    --output /data/output/detections.json \
    "$@"
