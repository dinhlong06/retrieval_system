#!/usr/bin/env bash
# Stage 2 (ising-calibration correction) driver, với tuỳ chọn tự chạy cả stage 1.
# NVIDIA_API_KEY đọc từ .env (xem .env.example) hoặc từ biến môi trường đã export.
#
# Cách dùng:
#   cp .env.example .env && vi .env       # điền NVIDIA_API_KEY, làm 1 lần
#   ./run_api.sh                          # mặc định: chỉ stage 2, dùng output/output_vietocr.json có sẵn
#   ./run_api.sh --full                   # tự chạy stage 1 (run_paddle.sh) rồi stage 2
#   PIPELINE=pipeline_a ./run_api.sh --full
#   ./run_api.sh --full --limit 60        # --full: forward mọi flag thừa cho run_paddle.sh (không cho stage 2)
#
# Output: mặc định output/output_hybrid.json (stage 2); --full còn tạo
# output/output_vietocr.json + output/output_paddle_origin.json (stage 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${NVIDIA_API_KEY:-}" && -f "$SCRIPT_DIR/.env" ]]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
fi
if [[ -z "${NVIDIA_API_KEY:-}" ]]; then
    echo "[x] NVIDIA_API_KEY chưa có -- điền vào .env (xem .env.example) hoặc export trước khi chạy." >&2
    exit 1
fi

FULL=0
if [[ "${1:-}" == "--full" ]]; then
    FULL=1
    shift
fi

if [[ "$FULL" -eq 1 ]]; then
    echo "== Stage 1: PaddleOCR (Docker, GPU) =="
    "$SCRIPT_DIR/run_paddle.sh" "$@"
else
    VIETOCR_OUTPUT="$SCRIPT_DIR/output/output_vietocr.json"
    if [[ ! -f "$VIETOCR_OUTPUT" ]]; then
        echo "[x] $VIETOCR_OUTPUT không tồn tại -- chạy ./run_paddle.sh trước, hoặc dùng ./run_api.sh --full." >&2
        exit 1
    fi
fi

echo "== Stage 2: ising-calibration correction (host) =="
(cd "$SCRIPT_DIR" && python3 run_correct.py)

echo "== Xong. Kết quả cuối: $SCRIPT_DIR/output/output_hybrid.json =="
