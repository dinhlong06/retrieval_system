#!/usr/bin/env bash
# So sánh Qwen3-VL-4B với baseline production có sẵn tại
# ../OCR/output_batch1/output_hybrid.json (không chạy lại PaddleOCR/VietOCR).
#
# Qwen có thể chạy ở 2 nơi:
#   A) vLLM server local (./run_vllm.sh start) -- cụm GPU dùng chung, hay bị tranh chấp.
#   B) vLLM server host trên Colab (xem qwen_vllm_colab_host.ipynb), gọi qua URL public.
#
#   ./run_vllm.sh start && ./run_compare.sh                      # (A) server local
#   VLLM_URL="https://xxxx.trycloudflare.com/v1" ./run_compare.sh # (B) server trên Colab
#
#   VIDEOS="L21_V001" ./run_compare.sh     # đổi danh sách video
#   ./run_compare.sh --limit 5             # mọi flag thừa forward cho compare.py
#
# ../OCR mount READ-ONLY: chỉ đọc ocr.loader (quét danh sách frame) và
# output_batch1/output_hybrid.json, không ghi/chạy gì trong đó.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE_NAME="ocr-v2-compare"
KEYFRAMES="$PROJECT_ROOT/dataset_batch1/keyframe/keyframes"
VIDEOS="${VIDEOS:-L21_V001 L21_V002 L21_V003}"
VLLM_PORT="${VLLM_PORT:-8811}"
# Mặc định gọi server local; đặt VLLM_URL="https://...trycloudflare.com/v1" để
# gọi server đang host trên Colab thay vì local.
VLLM_URL="${VLLM_URL:-http://localhost:$VLLM_PORT/v1}"

FRAME_DIRS=""
for v in $VIDEOS; do FRAME_DIRS="$FRAME_DIRS $KEYFRAMES/$v"; done

mkdir -p "$SCRIPT_DIR/output"

echo "== Build $IMAGE_NAME =="
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

# --network host chỉ cần khi gọi localhost của chính máy host (server local);
# với URL public (Colab/cloudflared) dùng network mặc định, DNS/internet bình thường.
is_local_url() { [[ "$VLLM_URL" == *"://localhost"* || "$VLLM_URL" == *"://127."* ]]; }

if [[ " $* " != *" --only prod "* ]]; then
    curl -sf "${VLLM_URL%/v1}/health" >/dev/null 2>&1 \
        || { echo "Không kết nối được vLLM server tại $VLLM_URL -- kiểm tra lại (local: ./run_vllm.sh start, hoặc Colab: notebook còn chạy?)"; exit 1; }
fi

NET_ARGS=(--network bridge)
is_local_url && NET_ARGS=(--network host)

docker run --rm "${NET_ARGS[@]}" \
    -v "$SCRIPT_DIR:/workspace/v2" \
    -v "$SCRIPT_DIR/../OCR:/workspace/OCR:ro" \
    -v "$KEYFRAMES:$KEYFRAMES:ro" \
    "$IMAGE_NAME" \
    --frames $FRAME_DIRS \
    --output-dir /workspace/v2/output \
    --vllm-url "$VLLM_URL" \
    "$@"
