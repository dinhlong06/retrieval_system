"""
compare.py -- Qwen3-VL (qua vLLM server, xem run_vllm.sh) vs baseline production
đã có sẵn tại ../OCR/output_batch1/output_hybrid.json.

Không tự chạy lại PaddleOCR+VietOCR: ../OCR/output_batch1/output_hybrid.json
đã là kết quả production ĐẦY ĐỦ 2 giai đoạn (PaddleOCR det + VietOCR rec, rồi
VLM ising-calibration phục hồi dấu) cho toàn bộ dataset_batch1, và phủ 100%
3 video mặc định (L21_V001/002/003, 855 frame) -- kiểm tra trực tiếp trước
khi viết file này. Dùng lại nó rẻ hơn nhiều so với chạy lại: khỏi tốn GPU
lần hai, khỏi cần cài paddleocr/vietocr vào image này.

output_vietocr.json (bản trung gian, TRƯỚC giai đoạn phục hồi dấu, ascii:
true trong config.yaml -- xem README) bị loại vì so nó với Qwen (vốn giữ
nguyên dấu) sẽ thiên vị Qwen một cách giả tạo.

../OCR chỉ còn cần cho `ocr.loader` (quét danh sách frame để biết cần gọi
Qwen trên file nào) -- không import paddle_engine nữa.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OCR_DIR = Path(__file__).resolve().parent.parent / "OCR"
sys.path.insert(0, str(OCR_DIR))

from ocr.formatter import save_output  # noqa: E402
from ocr.loader import load_frames  # noqa: E402


def _load(path: str) -> dict[str, dict]:
    p = Path(path)
    return {r["frame_id"]: r for r in json.loads(p.read_text(encoding="utf-8"))} if p.exists() else {}


def _run_qwen(frames, out_path, base_url, prompt, max_new_tokens, workers, checkpoint_every):
    from qwen_client import DEFAULT_PROMPT, QwenClient

    done = _load(out_path)
    todo = [(fid, p) for fid, p in frames if fid not in done]
    records = {fid: r for fid, r in done.items()}
    print(f"[qwen] {len(todo)} frame cần chạy ({len(done)} đã có từ checkpoint)", flush=True)

    client = QwenClient(base_url, prompt or DEFAULT_PROMPT, max_new_tokens)
    t0, n_done = time.time(), 0

    def call(frame_id, path):
        return frame_id, path, client.run(path)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(call, fid, p) for fid, p in todo]
        for fut in as_completed(futs):
            frame_id, path, text = fut.result()
            records[frame_id] = {"frame_id": frame_id, "path": path, "qwen": text}
            n_done += 1
            if n_done % checkpoint_every == 0:
                save_output([records[fid] for fid, _ in frames if fid in records], out_path)
                rate = n_done / (time.time() - t0)
                left = len(todo) - n_done
                print(f"[qwen] {n_done}/{len(todo)}  {rate:.2f} fps  eta {left/rate/60:.1f}m", flush=True)

    ordered = [records[fid] for fid, _ in frames]
    save_output(ordered, out_path)
    print(f"[qwen] xong sau {(time.time()-t0)/60:.1f}m -> {out_path}", flush=True)


def _extract_prod(frames, out_path, hybrid_path):
    """Lọc output_hybrid.json có sẵn xuống đúng bộ frame đang so sánh, không chạy lại engine nào."""
    wanted = {fid for fid, _ in frames}
    hybrid = _load(hybrid_path)
    missing = wanted - hybrid.keys()
    if missing:
        raise RuntimeError(
            f"{len(missing)} frame không có trong {hybrid_path} (ví dụ: {sorted(missing)[:5]}) -- "
            "giai đoạn 2 (VLM) của ../OCR có thể chưa chạy hết batch1 cho video này."
        )
    records = [{"frame_id": fid, "path": path, "prod": hybrid[fid]["texts"]} for fid, path in frames]
    save_output(records, out_path)
    print(f"[prod] lấy {len(records)} frame có sẵn từ {hybrid_path} -> {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen3-VL (vLLM) vs baseline production có sẵn")
    ap.add_argument("--frames", required=True, nargs="+", help="frame dirs (each scanned recursively)")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--limit", type=int, default=0, help="0 = every frame")
    ap.add_argument("--hybrid-json", default=None,
                     help="output_hybrid.json có sẵn để lấy baseline (mặc định ../OCR/output_batch1/output_hybrid.json)")
    ap.add_argument("--vllm-url", default="http://localhost:8811/v1")
    ap.add_argument("--prompt", default=None, help="override the OCR instruction sent to the VLM")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--qwen-workers", type=int, default=8, help="số request Qwen gửi song song")
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--only", choices=("qwen", "prod"), help="chỉ chạy một phần")
    args = ap.parse_args()

    frames = [f for d in args.frames for f in load_frames(d)]
    if args.limit:
        frames = frames[: args.limit]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[compare] {len(frames)} frames", flush=True)

    if args.only != "prod":
        print("[compare] pass 1/2 -- Qwen3-VL qua vLLM", flush=True)
        _run_qwen(frames, str(out / "qwen.json"), args.vllm_url, args.prompt,
                   args.max_new_tokens, args.qwen_workers, args.checkpoint_every)

    if args.only != "qwen":
        print("[compare] pass 2/2 -- lấy baseline có sẵn từ output_hybrid.json", flush=True)
        hybrid_path = args.hybrid_json or str(OCR_DIR / "output_batch1" / "output_hybrid.json")
        _extract_prod(frames, str(out / "prod.json"), hybrid_path)


if __name__ == "__main__":
    main()
