#!/usr/bin/env bash
# Layer 2 (CPU): ghép transcript (whisper.jsonl) vào shot (shots.jsonl).
# chạy thẳng bằng python3 hệ thống.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/cpu_map_transcript.py" \
    --shots_path "$SCRIPT_DIR/../../layer_1/shots.jsonl" \
    --whisper_path "$SCRIPT_DIR/../../layer_1/whisper.jsonl" \
    --output_path "$SCRIPT_DIR/shot_transcripts.jsonl"
