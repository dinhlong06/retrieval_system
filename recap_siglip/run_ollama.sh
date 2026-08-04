#!/usr/bin/env bash
# Quản lý Ollama server cho nhánh ReCap.
#
#   ./run_ollama.sh start   # chọn GPU rảnh, bật server, tạo model ghim VRAM, nạp
#   ./run_ollama.sh status  # model nào đang nạp, % trên GPU, giữ tới bao giờ
#   ./run_ollama.sh free    # nhả model khỏi VRAM, server vẫn chạy (trả GPU cho máy chung)
#   ./run_ollama.sh stop    # tắt hẳn server của mình
#
# `start` tự phát hiện và tự sửa khi model nạp hụt GPU; gọi lại `start` bất cứ
# lúc nào để ghim lại, không có subcommand riêng cho việc đó.
#
#   GPU_ID=5 ./run_ollama.sh start
#   BASE_MODEL=qwen3-vl:4b-instruct ./run_ollama.sh start
#
# 3 vấn đề đã đo được trên host này mà script này né:
#
# 1. Container gọi Ollama qua 10.0.2.3, KHÔNG phải 127.0.0.1/172.17.0.1 (docker
#    rootless + slirp, `--network host` không cấp netns thật) -> server phải
#    bind 0.0.0.0.
# 2. Ollama chốt tỉ lệ GPU/CPU LÚC LOAD và không tự sửa dù VRAM trống ra sau đó:
#    đo được 3.1/7.6GB -> 1.6 tok/s thay vì 76 tok/s, không báo lỗi gì.
# 3. `keep_alive` theo từng request GHI ĐÈ OLLAMA_KEEP_ALIVE của server; SDK
#    client mặc định gửi "10m" nên model tự unload sau 10 phút rảnh, và lần nạp
#    sau lại bốc thăm lại tỉ lệ ở vấn đề 2.
#
# Né (2)+(3): script tạo model dẫn xuất `<base>-gpu` từ Modelfile.gpu (PARAMETER
# num_gpu), Ollama từ chối nạp nếu không đủ VRAM thay vì âm thầm offload một phần
# — đo được ở GPU còn 4582MiB: bản gốc nạp 54% GPU im lặng, bản -gpu báo lỗi
# cứng. num_gpu không chặn được GPU quá chật (đo ratio=0.000 ở 1176MiB trống)
# nên vẫn giữ bước kiểm size_vram/size bên dưới. keep_alive luôn gửi -1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELFILE="${MODELFILE:-$SCRIPT_DIR/Modelfile.gpu}"
BASE_MODEL="${BASE_MODEL:-qwen3-vl:8b-instruct}"
MODEL="${MODEL:-$BASE_MODEL-gpu}"          # bản ghim num_gpu, script tự tạo nếu chưa có
NUM_GPU="${NUM_GPU:-999}"                  # 999 = ép toàn bộ layer lên GPU
OLLAMA_PORT="${OLLAMA_PORT:-11501}"
HOST_URL="http://127.0.0.1:$OLLAMA_PORT"
LOG_FILE="${LOG_FILE:-$HOME/ollama_$OLLAMA_PORT.log}"
MIN_VRAM_RATIO="${MIN_VRAM_RATIO:-0.95}"   # size_vram/size tối thiểu coi là "trọn trên GPU"
MIN_FREE_MIB="${MIN_FREE_MIB:-8500}"       # VRAM trống tối thiểu để chọn 1 GPU
ACTION="${1:-start}"

api() { curl -sf --max-time "${2:-15}" "$HOST_URL/$1"; }
up() { api api/tags 5 > /dev/null 2>&1; }
warm() { curl -sf --max-time 300 "$HOST_URL/api/chat" -d \
    "{\"model\":\"$MODEL\",\"stream\":false,\"keep_alive\":$1,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"options\":{\"num_predict\":1}}" > /dev/null; }
wait_unloaded() { while [ "$(api api/ps | jq '.models | length')" != "0" ]; do sleep 2; done; }

vram_info() {  # "<ratio> <used>/<total>GB <expires>" cho $MODEL, rỗng nếu chưa nạp
    api api/ps | jq -r --arg m "$MODEL" '
        .models[] | select(.name == $m) |
        "\(.size_vram/.size) \((.size_vram/1e9*10|round)/10)/\((.size/1e9*10|round)/10)GB \(.expires_at[0:19])"'
}
pinned() { awk -v r="${1%% *}" -v m="$MIN_VRAM_RATIO" 'BEGIN{exit !(r+0 >= m)}'; }

server_pid() {  # chỉ tìm server CỦA MÌNH, không đụng ollama của người khác
    for p in $(pgrep -u "$USER" -f 'ollama serve' 2>/dev/null || true); do
        if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q "OLLAMA_HOST=.*:$OLLAMA_PORT"; then
            echo "$p"; return 0
        fi
    done
    return 1
}
require_up() { up || { echo "LỖI: không có server ở $HOST_URL. Chạy: $0 start" >&2; exit 1; }; }

case "$ACTION" in
start)
    if up; then
        echo "== Server đã chạy sẵn ở $HOST_URL (pid $(server_pid || echo '?')) =="
    else
        GPU_ID="${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')}"
        FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')"
        if [ "$FREE" -lt "$MIN_FREE_MIB" ]; then
            echo "LỖI: GPU $GPU_ID chỉ trống ${FREE}MiB (< $MIN_FREE_MIB), model sẽ bị nạp hụt." >&2
            nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >&2
            echo "     Đặt GPU_ID=<index> để chọn tay, hoặc đợi GPU trống rồi chạy lại." >&2
            exit 1
        fi
        echo "== GPU $GPU_ID (${FREE}MiB trống) -> bind 0.0.0.0:$OLLAMA_PORT =="
        OLLAMA_HOST="0.0.0.0:$OLLAMA_PORT" OLLAMA_KEEP_ALIVE=-1 CUDA_VISIBLE_DEVICES="$GPU_ID" \
            nohup ollama serve > "$LOG_FILE" 2>&1 &
        for _ in $(seq 60); do up && break; sleep 1; done
        require_up
        echo "   log: $LOG_FILE"
    fi

    if ! api api/tags | grep -q "\"$BASE_MODEL\""; then
        echo "== Pull $BASE_MODEL =="
        OLLAMA_HOST="$HOST_URL" ollama pull "$BASE_MODEL"
    fi
    if ! api api/tags | grep -q "\"$MODEL\""; then
        echo "== Tạo $MODEL từ $MODELFILE (num_gpu $NUM_GPU) =="
        sed -e "s|__BASE_MODEL__|$BASE_MODEL|" -e "s|__NUM_GPU__|$NUM_GPU|" "$MODELFILE" \
            | OLLAMA_HOST="$HOST_URL" ollama create "$MODEL" -f /dev/stdin
    fi

    CAPS="$(curl -sf --max-time 10 "$HOST_URL/api/show" -d "{\"model\":\"$MODEL\"}" | jq -r '(.capabilities // []) | join(",")')"
    if [[ ",$CAPS," != *,vision,* ]]; then
        echo "LỖI: '$MODEL' không có capability 'vision' (có: $CAPS)." >&2
        exit 1
    fi
    echo "== capabilities: $CAPS =="

    echo "== Nạp + ghim model =="
    warm -1
    INFO="$(vram_info)"
    if ! pinned "$INFO"; then
        echo "   nạp hụt (${INFO#* }) -> nạp lại"
        warm 0; wait_unloaded; warm -1
        INFO="$(vram_info)"
    fi
    if ! pinned "$INFO"; then
        echo "LỖI: vẫn hụt (${INFO#* }). GPU đang bị chiếm; đợi rồi chạy lại '$0 start'." >&2
        exit 1
    fi
    echo "== SẴN SÀNG: $MODEL ${INFO#* } =="
    echo "   trên host    : --ollama-host $HOST_URL"
    echo "   trong docker : --ollama-host http://10.0.2.3:$OLLAMA_PORT"
    ;;

status)
    require_up
    echo "server pid : $(server_pid || echo 'không phải của bạn / không rõ')"
    api api/ps | jq -r '
        if (.models | length) == 0 then "model      : không có model nào đang nạp (VRAM đã nhả)"
        else .models[] |
            "model      : \(.name)\ntrên GPU   : \((.size_vram/1e9*10|round)/10)/\((.size/1e9*10|round)/10)GB (\((.size_vram/.size*100|round))%)\ngiữ tới    : \(.expires_at[0:19])"
        end'
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    ;;

free)
    require_up
    warm 0
    wait_unloaded  # nvidia-smi đọc ngay sau request thì VRAM chưa kịp nhả
    echo "== Đã nhả $MODEL khỏi VRAM (server vẫn chạy) =="
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    ;;

stop)
    PID="$(server_pid || true)"
    if [ -z "$PID" ]; then
        echo "Không tìm thấy server của bạn ở port $OLLAMA_PORT."
        exit 0
    fi
    kill "$PID"
    for _ in $(seq 10); do
        kill -0 "$PID" 2>/dev/null || break
        sleep 1
    done
    kill -0 "$PID" 2>/dev/null && kill -9 "$PID"
    echo "== Đã tắt server pid $PID =="
    ;;

*)
    echo "Dùng: $0 {start|status|free|stop}" >&2
    exit 1
    ;;
esac
