#!/usr/bin/env bash
# run.sh — chạy Keyframe Extractor bằng Docker (mặc định) hoặc trực tiếp host.
#
# Input thật lấy từ project root (chung với Layer 1):
#   video   : $PROJECT_ROOT/dataset/video
#   shots   : $PROJECT_ROOT/layer_1/shots.jsonl (1 file gộp mọi video)
# Layer 2 (runner.py) đòi 1 file shots/video, nên script tự tách shots.jsonl
# theo video_id ra dataset/shots/<video_id>.jsonl trước khi gọi cli.py.
#
# Dùng:
#   ./run.sh                         # pipeline_c, Docker
#   ./run.sh all                     # chạy cả 4 pipeline
#   ./run.sh pipeline_b --device cpu # truyền thêm flag CLI tuỳ ý
#   MODE=host ./run.sh pipeline_c    # chạy python3 cli.py thẳng, không Docker
#
# Mọi tham số sau tên pipeline được chuyển thẳng cho cli.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

PIPELINE="${1:-pipeline_c}"; shift || true
MODE="${MODE:-docker}"          # docker | host
IMAGE="keyframe-extractor:latest"

VIDEO_DIR_HOST="$PROJECT_ROOT/dataset/video"
SHOTS_SRC="$PROJECT_ROOT/layer_1/shots.jsonl"
SHOTS_SPLIT_DIR="$SCRIPT_DIR/dataset/shots"

# Tách shots.jsonl gộp thành 1 file/video (runner.py chỉ đọc được dạng này)
mkdir -p "$SHOTS_SPLIT_DIR"
if [[ -f "$SHOTS_SRC" ]]; then
  python3 - "$SHOTS_SRC" "$SHOTS_SPLIT_DIR" <<'PYEOF'
import collections, json, sys

src, out_dir = sys.argv[1], sys.argv[2]
by_video = collections.defaultdict(list)
with open(src, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            by_video[json.loads(line)["video_id"]].append(line)

for video_id, lines in by_video.items():
    with open(f"{out_dir}/{video_id}.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
PYEOF
else
  echo "[run.sh] WARNING: Không thấy $SHOTS_SRC — Layer 2 sẽ tự chunk video (không dùng shot thật)."
fi

if [[ "$MODE" == "host" ]]; then
  exec python3 cli.py --pipeline "$PIPELINE" \
    --video_dir "$VIDEO_DIR_HOST" \
    --shots_dir "$SHOTS_SPLIT_DIR" \
    "$@"
fi

# Máy dùng chung 8 GPU: ghim đúng 1 GPU (mặc định GPU rảnh nhất) thay vì --gpus all.
GPU_ID="${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')}"
echo "[run.sh] GPU $GPU_ID, DAKE_THREADS=${DAKE_THREADS:-4}"

exec docker run --rm --gpus "device=$GPU_ID" \
  -e PYTHONUNBUFFERED=1 \
  -e DAKE_THREADS="${DAKE_THREADS:-4}" \
  -e SHARD_ID="${SHARD_ID:-}" \
  -e CLAIMS_DIR="${CLAIMS_DIR:-}" \
  -v "$VIDEO_DIR_HOST:/app/dataset/raw_video:ro" \
  -v "$SHOTS_SPLIT_DIR:/app/dataset/shots" \
  -v "$SCRIPT_DIR/benchmark:/app/benchmark" \
  "$IMAGE" \
  --pipeline "$PIPELINE" \
  --video_dir dataset/raw_video \
  --shots_dir dataset/shots \
  "$@"
