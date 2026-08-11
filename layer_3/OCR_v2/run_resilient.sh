#!/usr/bin/env bash
# Chạy đủ 855 frame qua vLLM local, tự khởi động lại server + resume từ
# checkpoint nếu bị crash giữa chừng (cụm GPU dùng chung, tiến trình khác có
# thể chiếm VRAM bất cứ lúc nào trong lúc chạy, không chỉ lúc khởi động --
# xem README).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
TOTAL=855
MAX_ATTEMPTS=15

for attempt in $(seq 1 $MAX_ATTEMPTS); do
    done_count=$(python3 -c "import json,pathlib
p=pathlib.Path('output/qwen.json')
print(len(json.loads(p.read_text())) if p.exists() else 0)" 2>/dev/null || echo 0)
    echo "== Lần $attempt/$MAX_ATTEMPTS -- đã có $done_count/$TOTAL frame =="
    if [ "$done_count" -ge "$TOTAL" ]; then
        echo "ĐỦ $TOTAL frame, dừng."
        exit 0
    fi

    if ! curl -sf http://localhost:8811/health >/dev/null 2>&1; then
        echo "-- server chưa sống, khởi động lại --"
        ./run_vllm.sh start || { echo "Không khởi động được server, thử lại sau 20s"; sleep 20; continue; }
    fi

    # checkpoint-every thấp (10 thay vì mặc định 25): server có thể chết bất
    # cứ lúc nào giữa chừng, checkpoint dày hơn giảm số frame phải làm lại.
    VLLM_URL="http://localhost:8811/v1" ./run_compare.sh --only qwen --qwen-workers 4 --checkpoint-every 10
    sleep 3
done

echo "Hết $MAX_ATTEMPTS lần vẫn chưa đủ $TOTAL frame."
exit 1
