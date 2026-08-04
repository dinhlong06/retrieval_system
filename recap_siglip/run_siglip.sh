#!/usr/bin/env bash
# Build + chạy nhánh SigLIP của keyframe_pipeline trong Docker.
#
# Cách dùng:
#   ./run_siglip.sh                            # embed mọi video trong $PIPELINE
#   ./run_siglip.sh --overwrite                # ghi đè .npy đã có
#   ./run_siglip.sh --batch-size 64
#   PIPELINE=pipeline_a ./run_siglip.sh        # đổi pipeline nguồn
#   FRAMES_DIR=/path/khac ./run_siglip.sh      # bỏ qua PIPELINE
#
# Input : layer_2/Keyframe_Extracting/benchmark/<PIPELINE>/<VIDEO_ID>/<keyframe_id>.jpg
# Output: ./artifacts/siglip/<VIDEO_ID>.npy + <VIDEO_ID>_ids.json — cùng format với
#         embedding BEiT-3 mà layer 2 ghi ra, chỉ khác số chiều (1152 vs 1024).
#
# .model_cache/huggingface giữ weight SigLIP (~3,5 GB) trên NFS thay vì nhét vào
# image, vì ổ / của host gần đầy. Lần chạy đầu tải model, các lần sau dùng lại.
#
# Host này là GPU server dùng chung -> mặc định chọn 1 GPU đang rảnh nhất
# (free memory cao nhất) qua nvidia-smi, có thể override bằng biến GPU_ID.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="ai26-siglip"
PIPELINE="${PIPELINE:-pipeline_c}"
FRAMES_DIR="${FRAMES_DIR:-$PROJECT_ROOT/layer_2/Keyframe_Extracting/benchmark/$PIPELINE}"
OUTPUT_DIR="$SCRIPT_DIR/artifacts/siglip"
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/.model_cache/huggingface}"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

GPU_ID="${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')}"

echo "== Build image $IMAGE_NAME =="
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

echo "== Chạy SigLIP embedding trên GPU $GPU_ID =="
docker run --rm \
    --gpus "device=$GPU_ID" \
    --ipc=host \
    -v "$FRAMES_DIR:/data/frames:ro" \
    -v "$OUTPUT_DIR:/data/output" \
    -v "$CACHE_DIR:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    "$IMAGE_NAME" \
    siglip-dataset \
    --dataset-root /data/frames \
    --output-dir /data/output \
    --device cuda:0 \
    "$@"
