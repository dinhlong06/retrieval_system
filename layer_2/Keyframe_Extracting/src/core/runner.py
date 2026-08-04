"""
runner.py — Keyframe Benchmark Runner.

Điều phối toàn bộ quá trình benchmark:
  1. Đọc shots.json / shot.jsonl của từng video.
  2. Gọi extractor.extract() để lấy keyframe.
  3. Lưu ảnh keyframe vào thư mục output.
  4. Ghi keyframes.jsonl (metadata).
  5. Lưu embedding .npy và ids .json (nếu encoder trả về embedding).
  6. Tính và ghi statistics.json.
  7. Append vào benchmark_summary.csv.

Không biết chi tiết của pipeline — chỉ giao tiếp qua BaseKeyframeExtractor.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.core.interfaces import BaseKeyframeExtractor
from src.core.models import ShotRecord, ShotKeyframes, PipelineStatistics
from src.core.metrics import (
    compute_diversity_score,
    compute_coverage_score,
)


def _claim(video_id: str) -> bool:
    """Giành video giữa nhiều shard. O_CREAT|O_EXCL nguyên tử trên NFSv4 nên đúng
    một shard thắng. CLAIMS_DIR rỗng = chạy đơn, luôn thắng."""
    claims = os.environ.get("CLAIMS_DIR")
    if not claims:
        return True
    os.makedirs(claims, exist_ok=True)
    try:
        os.close(os.open(os.path.join(claims, video_id), os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        return False


class KeyframeBenchmarkRunner:
    """
    Runner điều phối toàn bộ quá trình benchmark Keyframe Extraction.

    Args:
        output_dir : Thư mục gốc chứa kết quả benchmark (mặc định "benchmark/").
    """

    def __init__(self, output_dir: str | Path = "benchmark"):
        self.output_dir = Path(output_dir)
        # Nhiều shard dùng chung output_dir (mỗi video một thư mục con nên không
        # đụng nhau), chỉ file summary là điểm ghi chung -> tách theo shard.
        self.summary_csv = self.output_dir / f"benchmark_summary{os.environ.get('SHARD_ID', '')}.csv"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        extractor: BaseKeyframeExtractor,
        video_paths: List[Path],
        shots_dir: Path,
        gt_dir: Optional[Path] = None,
    ) -> None:
        """
        Chạy benchmark cho một extractor trên tất cả video.

        Args:
            extractor   : Pipeline extractor (PipelineA/B/C/D).
            video_paths : Danh sách đường dẫn video .mp4.
            shots_dir   : Thư mục chứa shot.jsonl hoặc shots.json của từng video.
                          Cấu trúc: shots_dir/<video_stem>/shots.json
            gt_dir      : Thư mục chứa ground truth.
        """
        print(f"\n{'='*60}")
        print(f"  Benchmarking: {extractor.name.upper()}")
        print(f"{'='*60}")

        pipeline_out = self.output_dir / extractor.name
        pipeline_out.mkdir(parents=True, exist_ok=True)

        extractor.setup()

        all_stats: List[PipelineStatistics] = []

        for video_path in video_paths:
            if (pipeline_out / video_path.stem / "statistics.json").exists():
                print(f"[Runner] {video_path.name}: đã xong, bỏ qua.")
                continue
            if not _claim(video_path.stem):
                continue
            try:
                stats = self._run_single_video(
                    extractor=extractor,
                    video_path=video_path,
                    shots_dir=shots_dir,
                    pipeline_out=pipeline_out,
                    gt_dir=gt_dir,
                )
            except Exception as exc:
                print(f"[Runner] LỖI {video_path.name}: {type(exc).__name__}: {exc}")
                continue
            if stats:
                all_stats.append(stats)
                # Ghi ngay từng video: chạy 605 video mà crash giữa chừng thì
                # không mất toàn bộ summary.
                self._append_to_summary_csv([stats])

        extractor.teardown()
        print(f"\n[Runner] Finished: {extractor.name}. Results in {pipeline_out}\n")

    # ------------------------------------------------------------------
    # Private: single video processing
    # ------------------------------------------------------------------

    def _run_single_video(
        self,
        extractor: BaseKeyframeExtractor,
        video_path: Path,
        shots_dir: Path,
        pipeline_out: Path,
        gt_dir: Optional[Path] = None,
    ) -> Optional[PipelineStatistics]:
        """
        Xử lý một video: đọc shots, extract keyframes, lưu output, tính stats.

        Returns:
            PipelineStatistics nếu thành công, None nếu lỗi hoặc không có shot.
        """
        video_stem = video_path.stem
        print(f"\n[Runner] Processing {video_path.name} ...")

        # Đọc shots
        shots = self._load_shots(shots_dir, video_stem, video_path)
        if not shots:
            print(f"[Runner] WARNING: No shots found for {video_path.name}. Skipping.")
            return None

        if not self._can_decode(video_path):
            print(f"[Runner] WARNING: Không giải mã được {video_path.name} (codec không hỗ trợ, vd AV1). Skipping.")
            return None

        # Tạo thư mục output cho video này
        video_out = pipeline_out / video_stem
        video_out.mkdir(parents=True, exist_ok=True)
        # Không có statistics.json nghĩa là lần trước chạy dở: xoá keyframe cũ,
        # nếu không Layer 3 sẽ đọc lẫn ảnh của hai lần chạy.
        for stale in video_out.glob("*.jpg"):
            stale.unlink()

        # Track VRAM
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Đo thời gian
        t0 = time.time()
        shot_keyframes_list = extractor.extract(video_path, shots)
        elapsed = time.time() - t0

        peak_vram_gb = 0.0
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        # Lưu ảnh keyframe và điền image_path
        total_storage_bytes = self._save_keyframe_images(
            video_path, shot_keyframes_list, video_out
        )

        # Ghi keyframes.jsonl
        self._write_keyframes_jsonl(shot_keyframes_list, video_out)

        # Lưu embedding .npy + ids.json (tái dùng vector đã tính lúc Semantic Filter)
        kf_ids, kf_embeddings = self._collect_embeddings(shot_keyframes_list)
        self._save_embeddings(kf_ids, kf_embeddings, video_out, video_stem)

        # Tính stats
        stats = self._compute_statistics(
            extractor_name=extractor.name,
            video_id=video_stem,
            shots=shots,
            shot_keyframes_list=shot_keyframes_list,
            elapsed=elapsed,
            peak_vram_gb=peak_vram_gb,
            total_storage_bytes=total_storage_bytes,
            video_path=video_path,
            embeddings=kf_embeddings,
        )

        # Ghi statistics.json
        with open(video_out / "statistics.json", "w", encoding="utf-8") as f:
            json.dump(stats.__dict__, f, indent=4, ensure_ascii=False)
            
        # Đánh giá Ground Truth (nếu có)
        if gt_dir:
            video_gt_dir = gt_dir / video_stem
            if video_gt_dir.exists():
                print(f"[Runner] Evaluating Ground Truth from {video_gt_dir} ...")
                precision, recall, f1 = self._evaluate_gt(video_out, video_gt_dir)
                stats.precision = precision
                stats.recall = recall
                stats.f1_score = f1
                # Ghi đè lại statistics.json sau khi update
                with open(video_out / "statistics.json", "w", encoding="utf-8") as f:
                    json.dump(stats.__dict__, f, indent=4, ensure_ascii=False)

        print(
            f"[Runner] {video_path.name}: "
            f"{stats.total_keyframes} KFs, "
            f"{elapsed:.1f}s, "
            f"VRAM={peak_vram_gb:.2f}GB"
        )

        return stats

    # ------------------------------------------------------------------
    # Private: I/O helpers
    # ------------------------------------------------------------------

    def _load_shots(
        self,
        shots_dir: Path,
        video_stem: str,
        video_path: Path,
    ) -> List[ShotRecord]:
        """
        Đọc shots từ file JSON/JSONL.

        Tìm kiếm theo thứ tự:
          1. shots_dir/<video_stem>/shots.json  (shot detector benchmark output)
          2. shots_dir/<video_stem>/shot.jsonl  (streaming format)
          3. shots_dir/<video_stem>.json

        Trả về list ShotRecord. Nếu không tìm thấy, trả về shot giả toàn video.
        """
        candidates = [
            shots_dir / video_stem / "shots.json",
            shots_dir / video_stem / "shot.jsonl",
            shots_dir / f"{video_stem}.json",
            shots_dir / f"{video_stem}.jsonl",
        ]

        # Tìm file tồn tại đầu tiên
        shot_file = next((p for p in candidates if p.exists()), None)

        if shot_file is None:
            print(f"[Runner] No shots file found for {video_stem}. Chunking full video to prevent RAM OOM.")
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            shots = []
            chunk_size = 300  # Chia nhỏ mỗi shot tối đa 300 frames (~12s) để tránh tràn RAM
            for i in range(0, max(1, total), chunk_size):
                shots.append(ShotRecord(
                    video_id=video_stem,
                    shot_id=f"S{len(shots) + 1:04d}",
                    start_frame=i,
                    end_frame=min(i + chunk_size - 1, total - 1),
                    fps=fps,
                ))
            return shots

        shots: List[ShotRecord] = []
        with open(shot_file, encoding="utf-8") as f:
            # Hỗ trợ cả JSON array và JSONL (mỗi dòng 1 object)
            content = f.read().strip()
            if content.startswith("["):
                records = json.loads(content)
            else:
                records = [json.loads(line) for line in content.splitlines() if line.strip()]

        for i, rec in enumerate(records):
            shot_id = rec.get("shot_id", f"S{i + 1:04d}")
            # Hỗ trợ cả int shot_id và string shot_id
            if isinstance(shot_id, int):
                shot_id = f"S{shot_id:04d}"
            shots.append(ShotRecord(
                video_id=video_stem,
                shot_id=str(shot_id),
                start_frame=int(rec.get("start_frame", 0)),
                end_frame=int(rec.get("end_frame", 0)),
                fps=float(rec.get("fps", 25.0)),
            ))

        return shots

    def _can_decode(self, video_path: Path) -> bool:
        """Đọc thử frame đầu để phát hiện sớm codec không hỗ trợ (vd AV1), tránh loop hết mọi shot rồi mới lộ ra rỗng."""
        cap = cv2.VideoCapture(str(video_path))
        ret, _ = cap.read()
        cap.release()
        return ret

    def _collect_embeddings(
        self, shot_keyframes_list: List[ShotKeyframes]
    ) -> Tuple[List[str], List[np.ndarray]]:
        """Gom keyframe_id + embedding theo đúng thứ tự 1-1, xuyên suốt mọi shot của video."""
        ids: List[str] = []
        vectors: List[np.ndarray] = []
        for skf in shot_keyframes_list:
            for kf in skf.keyframes:
                if kf.embedding is not None:
                    ids.append(kf.keyframe_id)
                    vectors.append(kf.embedding)
        return ids, vectors

    def _save_embeddings(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        video_out: Path,
        video_stem: str,
    ) -> None:
        """Lưu {video_stem}.npy (N, D) float32 + {video_stem}_ids.json (list keyframe_id, cùng thứ tự dòng)."""
        if not vectors:
            return
        np.save(video_out / f"{video_stem}.npy", np.stack(vectors).astype(np.float32))
        with open(video_out / f"{video_stem}_ids.json", "w", encoding="utf-8") as f:
            json.dump(ids, f, ensure_ascii=False)

    def _save_keyframe_images(
        self,
        video_path: Path,
        shot_keyframes_list: List[ShotKeyframes],
        video_out: Path,
    ) -> int:
        """
        Lưu ảnh keyframe vào thư mục output và điền image_path.

        Cấu trúc output (phẳng, không lồng theo shot — để Layer 3 quét
        thư mục trực tiếp bằng frame_id = tên file, không cần đọc jsonl):
          video_out/
            <keyframe_id>.jpg    (vd: K01_V001_000000_kf0001.jpg)

        Args:
            video_path          : Đường dẫn video gốc.
            shot_keyframes_list : Danh sách ShotKeyframes (sẽ được update in-place).
            video_out           : Thư mục output của video.

        Returns:
            Tổng dung lượng ảnh đã lưu (bytes).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[Runner] Cannot open {video_path} for frame saving.")
            return 0

        total_bytes = 0

        for skf in shot_keyframes_list:
            for kf in skf.keyframes:
                cap.set(cv2.CAP_PROP_POS_FRAMES, kf.frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                img_filename = f"{kf.keyframe_id}.jpg"
                img_path = video_out / img_filename
                cv2.imwrite(str(img_path), frame)

                kf.image_path = img_filename
                total_bytes += img_path.stat().st_size if img_path.exists() else 0

        cap.release()
        return total_bytes

    def _write_keyframes_jsonl(
        self,
        shot_keyframes_list: List[ShotKeyframes],
        video_out: Path,
    ) -> None:
        """
        Ghi keyframes.jsonl — mỗi dòng là một keyframe record JSON.

        Format mỗi dòng:
          {"video_id": "...", "shot_id": "...", "keyframe_id": "...", ...}
        """
        jsonl_path = video_out / "keyframes.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for skf in shot_keyframes_list:
                for record in skf.to_records():
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Private: statistics
    # ------------------------------------------------------------------

    def _compute_statistics(
        self,
        extractor_name: str,
        video_id: str,
        shots: List[ShotRecord],
        shot_keyframes_list: List[ShotKeyframes],
        elapsed: float,
        peak_vram_gb: float,
        total_storage_bytes: int,
        video_path: Path,
        embeddings: List[np.ndarray],
    ) -> PipelineStatistics:
        """
        Tính PipelineStatistics từ kết quả extraction.

        Bao gồm: số KF, tốc độ xử lý, VRAM, storage, diversity, coverage.
        """
        total_kf = sum(len(skf.keyframes) for skf in shot_keyframes_list)
        avg_kf_per_shot = total_kf / max(len(shots), 1)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        processing_fps = total_frames / max(elapsed, 1e-6)
        storage_mb = total_storage_bytes / (1024 * 1024)

        # Tính coverage score (trung bình các shot)
        coverage_scores = []
        for shot, skf in zip(shots, shot_keyframes_list):
            kf_indices = [kf.frame_idx for kf in skf.keyframes]
            coverage_scores.append(compute_coverage_score(shot, kf_indices))
        avg_coverage = sum(coverage_scores) / max(len(coverage_scores), 1)

        return PipelineStatistics(
            pipeline=extractor_name,
            video_id=video_id,
            num_shots=len(shots),
            total_keyframes=total_kf,
            avg_keyframes_per_shot=avg_kf_per_shot,
            processing_time_sec=elapsed,
            processing_fps=processing_fps,
            peak_vram_gb=peak_vram_gb,
            total_storage_mb=storage_mb,
            diversity_score=compute_diversity_score(np.stack(embeddings)) if embeddings else 0.0,
            coverage_score=avg_coverage,
        )

    def _append_to_summary_csv(self, all_stats: List[PipelineStatistics]) -> None:
        """
        Append hàng thống kê vào benchmark_summary.csv.
        Tạo header nếu file chưa tồn tại.
        """
        if not all_stats:
            return

        file_exists = self.summary_csv.exists()
        fieldnames = list(all_stats[0].to_dict().keys())

        # Thử lưu file, nếu bị lock bởi Excel (PermissionError) thì báo lỗi và tạo file fallback
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(self.summary_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists and f.tell() == 0:
                        writer.writeheader()
                    for stats in all_stats:
                        writer.writerow(stats.to_dict())
                break  # Thành công thì thoát loop
            except PermissionError:
                if attempt < max_retries - 1:
                    print(f"\n[Runner] CẢNH BÁO: File {self.summary_csv.name} đang bị khóa (có thể do đang mở trong Excel).")
                    print("[Runner] Vui lòng đóng file. Đang thử lại trong 5 giây...")
                    time.sleep(5)
                else:
                    fallback_csv = self.output_dir / f"benchmark_summary_{int(time.time())}.csv"
                    print(f"\n[Runner] LỖI: Vẫn không thể ghi vào {self.summary_csv.name}.")
                    print(f"[Runner] Đang lưu tạm vào: {fallback_csv.name}")
                    with open(fallback_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for stats in all_stats:
                            writer.writerow(stats.to_dict())

    def _evaluate_gt(self, pred_dir: Path, gt_dir: Path) -> Tuple[float, float, float]:
        """
        Đánh giá precision, recall, f1 bằng cách encode ảnh Predicted và GT,
        rồi match bằng cosine similarity. Dùng MobileNetV3 cho nhanh.
        """
        from src.components.mobilenet_encoder import MobileNetEncoder
        from src.core.metrics import evaluate_with_ground_truth
        
        # Đọc ảnh Predicted (đã lưu ở output)
        pred_imgs = []
        for p in pred_dir.rglob("*.jpg"):
            img = cv2.imread(str(p))
            if img is not None:
                pred_imgs.append((len(pred_imgs), img))
                
        # Đọc ảnh GT (.png hoặc .jpg)
        gt_imgs = []
        for ext in ["*.png", "*.jpg"]:
            for p in gt_dir.rglob(ext):
                img = cv2.imread(str(p))
                if img is not None:
                    gt_imgs.append((len(gt_imgs), img))
                    
        if not pred_imgs or not gt_imgs:
            print(f"[Runner] Thiếu ảnh để evaluate: {len(pred_imgs)} Pred, {len(gt_imgs)} GT.")
            return 0.0, 0.0, 0.0
            
        print(f"[Runner] Đang encode {len(pred_imgs)} Pred và {len(gt_imgs)} GT images...")
        encoder = MobileNetEncoder(batch_size=64)
        encoder.load()
        _, pred_emb = encoder.encode_batch(pred_imgs)
        _, gt_emb = encoder.encode_batch(gt_imgs)
        encoder.unload()
        
        precision, recall, f1 = evaluate_with_ground_truth(pred_emb, gt_emb, threshold=0.85)
        print(f"[Runner] Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")
        return precision, recall, f1

