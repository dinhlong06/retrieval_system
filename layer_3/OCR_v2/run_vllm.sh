#!/usr/bin/env bash
# Khởi động vLLM OpenAI-compatible server cho qwen3-vl-4b-vietnamese-ocr-merged,
# giữ nguyên một GPU suốt vòng đời container thay vì nạp/xả model mỗi lần chạy
# như hướng transformers cũ -- đỡ tranh chấp VRAM với tiến trình khác trên
# cụm dùng chung, vì chỉ giành GPU một lần lúc start thay vì mỗi frame.
#
#   ./run_vllm.sh start   # build + chạy server nền, healthcheck xong mới trả
#   ./run_vllm.sh stop
#   ./run_vllm.sh logs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="ocr-v2-vllm"
CONTAINER_NAME="ocr-v2-vllm-server"
MODEL_DIR="$SCRIPT_DIR/models/qwen3-vl-4b-vi-ocr"
PORT="${VLLM_PORT:-8811}"

cmd="${1:-start}"

case "$cmd" in
stop)
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    exit 0
    ;;
logs)
    docker logs -f "$CONTAINER_NAME"
    exit 0
    ;;
start) ;;
*) echo "dùng: $0 {start|stop|logs}"; exit 1 ;;
esac

[ -d "$MODEL_DIR" ] || { echo "Chưa có model, chạy: hf download huypl53/qwen3-vl-4b-vietnamese-ocr-merged --local-dir $MODEL_DIR"; exit 1; }
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "== Build $IMAGE_NAME =="
docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile.vllm" "$SCRIPT_DIR"

# Cụm dùng chung: card rảnh nhất thường chỉ còn 2-5GB, không đủ cho fp16 (cần
# ~8.3GB) -- nạp NF4 qua bitsandbytes của chính vLLM (khác hẳn transformers+bnb
# đã thử trước đó: vLLM giữ model thường trực + continuous batching, không
# phải nạp lại mỗi request). NF4 ~2.8GB, cộng KV cache + vision tower activation.
# Đo thật (lần chạy trước): weight NF4 chiếm 3.15GiB -- KV_MARGIN_MIB=300 cũ
# để "Available KV cache memory" ra ÂM (-0.6GiB) vì ngân sách utilization
# chỉ vừa đủ chứa weight, không còn gì cho KV cache/activation. Nâng margin
# lên rõ ràng dư, không phải OOM tranh chấp.
MODEL_MIB=3200
KV_MARGIN_MIB=1800
NEED_MIB=$((MODEL_MIB + KV_MARGIN_MIB))
QUANT_ARGS=(--quantization bitsandbytes --load-format bitsandbytes)

try_gpu() {
    local gpu_id="$1" free_mib="$2"
    local total_mib util
    total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" | tr -d ' ')
    util=$(python3 -c "print(min(0.85, $NEED_MIB / $total_mib))")
    echo "== Thử GPU $gpu_id: ${free_mib}MiB trống -> --gpu-memory-utilization=$util (~${NEED_MIB}MiB) =="

    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker run -d --name "$CONTAINER_NAME" \
        --gpus "device=$gpu_id" \
        --shm-size=8g \
        -p "$PORT:8000" \
        -v "$MODEL_DIR:/model:ro" \
        -v "$SCRIPT_DIR/cache/hf:/root/.cache/huggingface" \
        "$IMAGE_NAME" \
        --model /model --served-model-name qwen3-vl-vi-ocr \
        --dtype float16 \
        "${QUANT_ARGS[@]}" \
        --gpu-memory-utilization "$util" \
        --max-model-len 2048 \
        --enforce-eager \
        --limit-mm-per-prompt '{"image": 1}' >/dev/null

    for _ in $(seq 1 90); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
            echo "vLLM server sẵn sàng tại http://localhost:$PORT (GPU $gpu_id)"
            return 0
        fi
        if ! docker ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
            echo "-- GPU $gpu_id: container thoát sớm (khả năng OOM do bị chiếm giữa chừng) --"
            docker logs "$CONTAINER_NAME" 2>&1 | tail -15
            return 1
        fi
        sleep 10
    done
    echo "-- GPU $gpu_id: timeout 15 phút chờ health, coi như thất bại --"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -30
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    return 1
}

# Thử tối đa 20 lần (~10 phút), mỗi lần chọn lại GPU rảnh nhất TẠI THỜI ĐIỂM
# ĐÓ (không dùng danh sách cũ) vì đây chính là nguồn OOM ban đầu.
for attempt in $(seq 1 90); do
    read -r GPU_ID FREE_MIB <<< "$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t',' -k2 -n -r | head -1 | tr -d ' ' | tr ',' ' ')"
    if [ "$FREE_MIB" -lt "$NEED_MIB" ]; then
        echo "Lần thử $attempt: GPU rảnh nhất (GPU $GPU_ID) chỉ còn ${FREE_MIB}MiB, cần ${NEED_MIB}MiB. Chờ 30s rồi thử lại."
        sleep 30
        continue
    fi
    if try_gpu "$GPU_ID" "$FREE_MIB"; then
        exit 0
    fi
    echo "Lần thử $attempt thất bại, thử GPU khác..."
done

echo "Hết 4 lần thử vẫn không khởi động được vLLM server. Xem log GPU ở trên."
exit 1
